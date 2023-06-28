#!/usr/bin/env python
#############################################################################
#
# This is the bulk of the logic for the gradio demo. You use it for whatever 
# you want. Credit would be nice but w/e
#
# You can also run it on an image from the cli
#
# TODO:
#
# 1. rework the classes that just wrap Dict and List to extend them
# 2. cleanup all the to_dict madness
# 3. convert the print calls to use the logging
# 4. add a proper creative commons license 
# 5. cleanup string constants
# 6. replace custom code with libraries like for OBJ
# 
#############################################################################

import cv2
import json
import logging
import mediapipe as mp
import numpy as np
import os
import sys
import torch

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image, ImageDraw
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from typing import List, Mapping, Optional, Tuple, Union, Dict, Type

from utils import colorize
from quads import QUADS

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

NumpyImage = Type[np.ndarray]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HF_HACK = True

class Point3:
    def __init__(self, values:List[float]=3*[0] ):
        self.values = values

    @property
    def x(self):
        return self.values[0]

    @property
    def y(self):
        return self.values[1]

    @property
    def z(self):
        return self.values[2]

    def to_dict(self):
        return {'x':self.x,'y':self.y,'z':self.z}


class TextureCoordinate:
    def __init__(self, values:List[float]=2*[0] ):
        self.values = values

    @property
    def u(self):
        return self.values[0]

    @property
    def v(self):
        return self.values[1]

    def to_dict(self):
        return {'u':self.u,'v':self.v}

class PixelCoordinate:
    def __init__(self, values:List[int]=2*[0] ):
        self.values = values

    @property
    def x(self):
        return self.values[0]

    @property
    def y(self):
        return self.values[1]

    def to_dict(self):
        return {'x':self.x,'y':self.y}

class DepthMap:
    MEDIA_PIPE = 'mediapipe'
    def __init__(self, values:Dict[str,float]={'og':0} ):
        self.values = values

    def to_dict(self):
        return self.values

class DepthMapping:
    def __init__(self, weight:float=1, lo:float=+np.inf, hi:float=-np.inf, toLo:float=0, toHi:float=1):
        self.weight = weight
        self.lo = lo
        self.hi = hi
        self.toLo = toLo
        self.toHi = toHi
        self.diff = 1
        self.toDiff = 1
        self.update()

    def reset(self):
        self.lo = +np.inf
        self.hi = -np.inf

    def track(self,value):
        self.lo = min(self.lo,value)
        self.hi = max(self.hi,value)

    def update(self):
        self.diff = self.hi - self.lo
        self.toDiff = self.toHi - self.toLo
        return self

    def translate(self,value):
        if not self.diff == 0:
            value = ( value - self.lo ) / self.diff
            value = self.toLo + value * self.toDiff
        value = value * self.weight
        return value

    def to_dict(self):
        return {
            'weight' : self.weight,
            'lo'     : self.lo,
            'hi'     : self.hi,
            'toLo'   : self.toLo,
            'toHi'   : self.toHi,
            'diff'   : self.diff,
            'toDiff' : self.toDiff,
        }


class WeightMap:
    def __init__(self, values:Dict[str,DepthMapping]=None):
        if values is None:
            self.values = {DepthMap.MEDIA_PIPE:DepthMapping()} 
        else:
            self.values = values

    def set(self,key:str,depthMapping:DepthMapping):
        self.values[key] = depthMapping

    def totally(self,name:str):
        if not name in self.values:
            raise Exception( f'no weight for {k} in {self.to_dict()}' )
        for depthMapping in self.values.values():
            depthMapping.weight = 0
        self.values[ name ].weight = 1

    def saveWeights(self)->Dict[str,float]:
        return {k:v.weight for k,v in self.values.items()}

    def loadWeights(self,weights:Dict[str,float]):
        for k,weight in weights.items():
            if k in self.values:
                self.values[ k ].weight = weight
            else:
                raise Exception( f'no weight for {k} in {self.to_dict()}' )

    def to_dict(self):
        return {k:dm.to_dict() for k,dm in self.values.items()}
        return self.values

class MeshPoint:
    def __init__(self, 
        position:Point3 = Point3(),
        color:Point3 = Point3(),
        textureCoordinate:TextureCoordinate = TextureCoordinate(), 
        pixelCoordinate:PixelCoordinate = PixelCoordinate(), 
        depthMap:DepthMap = None,
    ):
        self.position = position
        self.color = color
        self.textureCoordinate = textureCoordinate
        self.pixelCoordinate = pixelCoordinate

        if depthMap is None:
            self.depthMap = DepthMap({DepthMap.MEDIA_PIPE:position.values[2]})
        else:
            self.depthMap = depthMap

    def to_dict(self):
        derp = {
            'position'          : self.position.to_dict(),
            'color'             : self.color.to_dict(),
            'textureCoordinate' : self.textureCoordinate.to_dict(),
            'pixelCoordinate'   : self.pixelCoordinate.to_dict(),
        }
        if not self.depthMap is None:
            derp[ 'depthMap' ] = self.depthMap.to_dict()
        return derp

    def weighDepth(self, weightMap:WeightMap = WeightMap()):
        total_sum = sum([dm.weight for dm in weightMap.values.values()])
        tmp = 0
        for key, depthMapping in weightMap.values.items():
            if key in self.depthMap.values:
                tmp = tmp + depthMapping.translate( self.depthMap.values[ key ] )
            else:
                raise Exception(f'{key} from weightMap not in depthMap')
        tmp = tmp / total_sum 
        #print( f'depthMap: {json.dumps(self.depthMap.to_dict())} -> {tmp}') # spam!!!
        self.position.values[2] = tmp 

    def mapLandMark(self, mediaMesh:'MediaMesh', landmark: landmark_pb2.NormalizedLandmark) -> 'MeshPoint':
        x, y = _normalized_to_pixel_coordinates(landmark.x,landmark.y,mediaMesh.width,mediaMesh.height)

        #position = [landmark.x * mediaMesh.ratio, landmark.y, landmark.z]
        #position = [landmark.x * mediaMesh.ratio, landmark.y, landmark.z]
        position = [v * mediaMesh.scale[i] for i,v in enumerate([landmark.x, landmark.y, landmark.z])]

        self.position = Point3(position)

        #self.position = Point3([landmark.x * mediaMesh.ratio, landmark.y, landmark.z])
        self.color = Point3([value / 255 for value in mediaMesh.image[y,x]])
        self.textureCoordinate = TextureCoordinate([x/mediaMesh.width,1-y/mediaMesh.height] )
        self.pixelCoordinate = PixelCoordinate([x,y])
        self.depthMap = DepthMap({DepthMap.MEDIA_PIPE:self.position.z})
        return self

    def toObj(self, lines:List[str], hf_hack:bool=HF_HACK):
        lines.append( "v  " + " ".join(map(str, self.position.values + self.color.values)) )
        lines.append( "vt " + " ".join(map(str, self.textureCoordinate.values ) ) )

# IMPORTANT! MeshFace uses 1 based indices, not 0 based!!!!
class MeshFace:
    def __init__(self,indices:List[int]=None,normal:Point3=Point3()):
        self.indices = indices
        self.normal = normal

    def calculateNormal(self,meshPoints:List[MeshPoint]):
        if self.indices is None:
            raise Exception('indices is junk')
        if meshPoints is None:
            raise Exception('meshPoints is junk')
        if len(self.indices)<3:
            raise Exception('need at least 3 points')

        points = [meshPoints[index-1] for index in self.indices[:3]]
        npz = [np.array(point.position.values) for point in points]

        v1 = npz[1] - npz[0]
        v2 = npz[2] - npz[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        self.normal = Point3( normal.tolist() )

    def toObj(self, lines:List[str], index:int, hf_hack:bool=HF_HACK):
        lines.append( "vn " + " ".join([str(value) for value in self.normal.values]) )
        face_uv = "f " + " ".join([f'{vertex}/{vertex}/{index}' for vertex in self.indices])
        face_un = "f " + " ".join([str(vertex) for vertex in self.indices])
        if hf_hack:
            lines.append( f'#{face_uv}' )
            lines.append( f'{face_un}' )
        else:
            lines.append( face_uv )

class DepthSource:
    def __init__(self, name:str=None):
        self.name = name 
        self.mediaMesh = None
        self.depth:NumpyImage = None
        self.gray:NumpyImage = None

    def mapDepth(self, mediaMesh:'MediaMesh', depthMapping:DepthMapping=None) -> 'DepthSource':
        return self
    
    def _addDepth(self, mediaMesh:'MediaMesh', depthMapping:DepthMapping=None) -> 'DepthSource':
        self.gray = colorize(self.depth, cmap='gray_r')
        self.mediaMesh = mediaMesh

        for meshPoint in mediaMesh.points:
            depth = self.depth[meshPoint.pixelCoordinate.y,meshPoint.pixelCoordinate.x]
            #depth = -depth # lazy conversion from depth to position
            meshPoint.depthMap.values[ self.name ] = float( depth )

        mediaMesh.weightMap.set( self.name, self.createDepthMapping(depthMapping) )

        self.gray = mediaMesh.drawGrayMesh(self.name,True)
        return self

    # note: if depthMapping is passed in, the hi and lo will be reset
    def createDepthMapping(self,depthMapping:DepthMapping=None) -> DepthMapping:
        if depthMapping is None:
            depthMapping = DepthMapping()
        depthMapping.reset()
        if not self.depth is None:
            for meshPoint in self.mediaMesh.points:
                depth = self.depth[meshPoint.pixelCoordinate.y,meshPoint.pixelCoordinate.x]
                depthMapping.track(float(depth))
        return depthMapping.update()

class ZoeDepthSource( DepthSource ):
    NAME = 'zoe'

    def __init__(self):
        super().__init__(ZoeDepthSource.NAME)
        self.model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

    def mapDepth(self, mediaMesh:'MediaMesh', depthMapping:DepthMapping=None) -> 'DepthSource':
        self.depth = 1.-self.model.infer_pil(mediaMesh.image)
        return self._addDepth(mediaMesh, depthMapping)

class MidasDepthSource( DepthSource ):
    NAME = 'midas'

    def __init__(self):
        super().__init__(MidasDepthSource.NAME)
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    def mapDepth(self, mediaMesh:'MediaMesh', depthMapping:DepthMapping=None) -> 'DepthSource':
        img = Image.fromarray(mediaMesh.image)

        encoding = self.feature_extractor(img, return_tensors="pt")
        with torch.no_grad():
           outputs = self.model(**encoding)
           predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        self.depth = prediction.cpu().numpy()
        return self._addDepth(mediaMesh, depthMapping)

#############################################################################
#
# A MediaMesh has:
#
# 1. an input image
# 2. the first landmark found
# 3. a MeshPoint for each point 
#
#
#
#############################################################################
class MediaMesh:
    LOG = logging.getLogger(__name__)
    COMBINED = 'combined'

    def __init__(self, scale:List[int]=[-1,-1,-1], weightMap:WeightMap = None, image:NumpyImage = None, annotated:NumpyImage = None, points:List[MeshPoint] = None):
        self.scale = scale
        if weightMap is None:
            self.weightMap = WeightMap()
        else:
            self.weightMap = weightMap
        self.image = image
        self.annotated = annotated
        self.points = points
        self.meshes = {}
        self.depthSources = {}

    # after this call, instance variables for image, annotated and points should be set
    def detect(self, image:NumpyImage, min_detection_confidence:float = .5) -> 'MediaMesh':
        self.image = image
        self.annotated = image.copy()
        self.points = None

        self.width  = image.shape[1]
        self.height = image.shape[0]
        self.ratio  = self.width / self.height

        self.scale[0] = self.ratio

        first = True # just do the first face for now

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raise Exception( 'no faces found' )

            for landmarks in results.multi_face_landmarks:
                if first:
                    self.points = self.mapLandMarks(landmarks)
                    first = False
                self.drawLandMarks(self.annotated, landmarks)
        
        self.gray = self.drawGrayMesh()
        self.weightMap.set( DepthMap.MEDIA_PIPE, self.createDepthMapping() )

        return self

    def drawLandMarks(self, image:NumpyImage, landmarks: landmark_pb2.NormalizedLandmarkList):
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

    def mapLandMarks(self, landmarks: landmark_pb2.NormalizedLandmarkList) -> List[MeshPoint]:
        points = []
        for landmark in landmarks.landmark:
            point = MeshPoint().mapLandMark(self, landmark)
            points.append( point )
        return self.centerPoints(points)

    def centerPoints(self,points:List[MeshPoint]=None) -> List[MeshPoint]:
        if points is None:
            points = self.points

        mins = [+np.inf] * 3
        maxs = [-np.inf] * 3

        for point in points:
            for dimension,value in enumerate( point.position.values ):
                mins[dimension] = min(mins[dimension],value)
                maxs[dimension] = max(maxs[dimension],value)

        mids = [(min_val + max_val) / 2 for min_val, max_val in zip(mins, maxs)]
        for point in points:
            point.position.values = [(val-mid) for val, mid in zip(point.position.values,mids)]

        print( f'mins: {mins}' )
        print( f'mids: {mids}' )
        print( f'maxs: {maxs}' )

        return points

    def createDepthMapping(self,depthMapping:DepthMapping=None) -> DepthMapping:
        if depthMapping is None:
            depthMapping = DepthMapping()
        for point in self.points:
            depthMapping.track(point.position.z)
        return depthMapping.update()

    def drawGrayMesh(self, source:str=DepthMap.MEDIA_PIPE, invert:bool=False):
        image = Image.new("RGB", (self.width, self.height), (88,13,33))
        draw = ImageDraw.Draw(image)

        minZ = np.inf
        maxZ = -np.inf

        depths = []

        for point in self.points:
            depth = point.depthMap.values[source]
            depths.append( depth )
            minZ = min( minZ, depth )
            maxZ = max( maxZ, depth )

        difZ = maxZ - minZ
        if 0 == difZ:
            difZ = 1

        depths = [(depth-minZ)/difZ for depth in depths]

        for quad in QUADS:
            points = [tuple(self.points[index-1].pixelCoordinate.values) for index in quad]
            colors = [tuple(3*[int(255*depths[index-1])]) for index in quad]
            color = int(np.average(colors))
            if invert:
                color = 255 - color
            draw.polygon(points, fill=tuple(3*[color]))
            #draw.polygon(points, fill=colors) # sadly this does not work

        return np.asarray(image)

    # the obj is based on the current weightMap
    def toObj(self, name:str='sweet', hf_hack:bool=HF_HACK):
        print( '-----------------------------------------------------------------------------' )

        obj = [f'o {name}Mesh']
        mtl = f'newmtl {name}Material\nmap_Kd {name}.png\n'

        c = '#' if hf_hack else ''
        obj.append( f'{c}mtllib {name}.mtl' )

        obj.append( f'##################################################################' )
        obj.append( f'# to bring into blender with uvs:' )
        obj.append( f'# put the following 2 lines into {name}.mtl uncommented' )
        obj.append( f'#newmtl {name}Material' )
        obj.append( f'#map_Kd {name}.png' )
        obj.append( f'# remove lines from this file starting with "f  "' )
        obj.append( f'# uncomment the lines that start with "#f "' )
        obj.append( f'##################################################################' )

        for key, depthMapping in self.weightMap.values.items():
            depthMapping.update()
            print( f'{name}.{key} -> {depthMapping.to_dict()}' )

        for point in self.points:
            point.weighDepth(self.weightMap)

        self.centerPoints()

        for point in self.points:
            point.toObj(obj,hf_hack)

        obj.append( f'usemtl {name}Material' )

        index = 0
        for quad in QUADS:
            index = 1 + index
            face = MeshFace(quad)
            face.calculateNormal(self.points)
            face.toObj(obj, index, hf_hack)  

        obj.append( f'##################################################################' )
        obj.append( f'# EOF' )
        obj.append( f'##################################################################' )

        print( '-----------------------------------------------------------------------------' )

        return obj,mtl

    def to_dict(self):
        return {
            'width'     : self.width,
            'height'    : self.height,
            'ratio'     : self.ratio,
            'weightMap' : {key: value.to_dict() for key, value in self.weightMap.values.items()},
            'points'    : [point.to_dict() for point in self.points]
        }

    # should be called after demoSetup and detect
    def singleSourceMesh(self,name:str, hf_hack:bool=HF_HACK):
        before = self.weightMap.saveWeights() # push
        self.weightMap.totally(name)
        obj,mtl = self.toObj(name)
        self.weightMap.loadWeights( before ) # pop
        return obj,mtl

    # should be called after demoSetup and detect
    def meshmerizing(self,hf_hack:bool=HF_HACK):
        for depthSource in self.depthSources:
            depthSource.mapDepth(self,self.weightMap.values[depthSource.name])

        obj,mtl = self.toObj(MediaMesh.COMBINED)
        self.meshes = {MediaMesh.COMBINED:(obj,mtl)}

        for source in self.depthSources:
            self.meshes[ source.name ] = (self.singleSourceMesh(source.name))

        self.meshes[DepthMap.MEDIA_PIPE] = (self.singleSourceMesh(DepthMap.MEDIA_PIPE))

        return self.meshes

    def demoSetup(self) -> 'MediaMesh':
        self.depthSources = [ ZoeDepthSource(), MidasDepthSource() ]

        for depthSource in self.depthSources:
            self.weightMap.set( depthSource.name, depthSource.createDepthMapping() )

        # observationally
        self.weightMap.values[ ZoeDepthSource.NAME   ].toHi   = 1.77
        self.weightMap.values[ MidasDepthSource.NAME ].toHi   = 2.55
        self.weightMap.values[ ZoeDepthSource.NAME   ].weight = 1.00
        self.weightMap.values[ MidasDepthSource.NAME ].weight = 0.22

        return self

    def main(self):
        if not 2 == len(sys.argv):
            raise Exception( 'usage: MediaMesh.py <image filename>' )
        mediaMesh = MediaMesh().demoSetup()
        mediaMesh.detect(cv2.imread( sys.argv[1] ) ) 

        for name,mesh in mediaMesh.meshmerizing().items():
            obj = mesh[0]
            mtl = mesh[1]
            with open(f"{name}.obj", "w") as file:
                file.write( '\n'.join(obj) )
            with open(f"{name}.mtl", "w") as file:
                file.write( mtl )

        cv2.imwrite( 'mesh.png', mediaMesh.annotated )
        cv2.imwrite( 'mpg.png',  mediaMesh.gray )
        for source in mediaMesh.depthSources:
            cv2.imwrite( f'{source.name}.png', source.gray )

        with open("mesh.json", "w") as file:
            json.dump(mediaMesh.to_dict(), file, indent=4)

if __name__ == "__main__":
    MediaMesh().main()

# EOF
#############################################################################
