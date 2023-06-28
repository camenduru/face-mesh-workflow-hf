########################################################################################
import gradio as gr

import cv2
import matplotlib
import matplotlib.cm
import mediapipe as mp
import numpy as np
import os
import struct
import tempfile
import torch

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image
from quads import QUADS
from typing import List, Mapping, Optional, Tuple, Union
from utils import colorize, get_most_recent_subdirectory

class face_image_to_face_mesh:
    def __init__(self):
        self.zoe_me = True
        self.uvwrap = not True

    def demo(self):
        if self.zoe_me:
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.zoe = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

        demo = gr.Blocks(css=self.css(), cache_examples=True)
        with demo:
            gr.Markdown(self.header())

            with gr.Row():
                with gr.Column():
                    upload_image = gr.Image(label="Input image", type="numpy", source="upload")
                    self.temp_dir = get_most_recent_subdirectory( upload_image.DEFAULT_TEMP_DIR )
                    print( f'The temp_dir is {self.temp_dir}' )

                    gr.Examples( examples=[
                        'examples/blonde-00081-399357008.png',
                        'examples/dude-00110-1227390728.png',
                        'examples/granny-00056-1867315302.png',
                        'examples/tuffie-00039-499759385.png',
                        'examples/character.png',
                    ], inputs=[upload_image] )
                    upload_image_btn = gr.Button(value="Detect faces")
                    if self.zoe_me:
                        with gr.Group():
                            zoe_scale = gr.Slider(label="Mix the ZoeDepth with the MediaPipe Depth", value=1, minimum=0, maximum=1, step=.01)
                            flat_scale = gr.Slider(label="Depth scale, smaller is flatter and possibly more flattering", value=1, minimum=0, maximum=1, step=.01)
                            min_detection_confidence = gr.Slider(label="Mininum face detection confidence", value=.5, minimum=0, maximum=1.0, step=0.01)
                    else:
                        use_zoe = False
                        zoe_scale = 0
                    with gr.Group():
                        gr.Markdown(self.footer())

                with gr.Column():
                    with gr.Group():
                        output_mesh = gr.Model3D(clear_color=3*[0],  label="3D Model",elem_id='mesh-display-output')
                        output_image = gr.Image(label="Output image",elem_id='img-display-output')
                        depth_image = gr.Image(label="Depth image",elem_id='img-display-output')
                        num_faces_detected = gr.Number(label="Number of faces detected", value=0)

            upload_image_btn.click(
                fn=self.detect, 
                inputs=[upload_image, min_detection_confidence,zoe_scale,flat_scale], 
                outputs=[output_mesh, output_image, depth_image, num_faces_detected]
            )
        demo.launch()


    def detect(self, image, min_detection_confidence, zoe_scale, flat_scale):
        width  = image.shape[1]
        height = image.shape[0]
        ratio  = width / height

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
            
        mesh = "examples/converted/in-granny.obj"

        if self.zoe_me and 0 < zoe_scale:
            depth = self.zoe.infer_pil(image)
            idepth = colorize(depth, cmap='gray_r')
        else:
            depth = None
            idepth = image

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return mesh, image, idepth, 0

            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                (mesh,mtl,png) = self.toObj(image=image, width=width, height=height, ratio=ratio, landmark_list=face_landmarks, depth=depth, zoe_scale=zoe_scale, flat_scale=flat_scale)

                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            return mesh, annotated_image, idepth, 1

    def toObj( self, image: np.ndarray, width:int, height:int, ratio: float, landmark_list: landmark_pb2.NormalizedLandmarkList, depth: np.ndarray, zoe_scale: float, flat_scale: float):
        print( f'you have such pretty hair', self.temp_dir )

        hf_hack = True
        if hf_hack:
            obj_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
            mtl_file = tempfile.NamedTemporaryFile(suffix='.mtl', delete=False)
            png_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        else:
            obj_file = tempfile.NamedTemporaryFile(suffix='.obj', dir=self.temp_dir, delete=False)
            mtl_file = tempfile.NamedTemporaryFile(suffix='.mtl', dir=self.temp_dir, delete=False)
            png_file = tempfile.NamedTemporaryFile(suffix='.png', dir=self.temp_dir, delete=False)

        ############################################
        (points,coordinates,colors) = self.landmarksToPoints( image, width, height, ratio, landmark_list, depth, zoe_scale, flat_scale )
        ############################################

        lines = []

        lines.append( f'o MyMesh' )

        if hf_hack:
            # the 'file=' is a gradio hack
            lines.append( f'#mtllib file={mtl_file.name}' )
        else:
            # the 'file=' is a gradio hack
            lines.append( f'mtllib file={mtl_file.name}' )

        for index, point in enumerate(points):
            color = colors[index]
            scaled_color = [value / 255 for value in color]  # Scale colors down to 0-1 range
            flipped = [-value for value in point]
            flipped[ 0 ] = -flipped[ 0 ]
            lines.append( "v " + " ".join(map(str, flipped + color)) )

        for coordinate in coordinates:
            lines.append( "vt " + " ".join([str(value) for value in coordinate]) )

        for quad in QUADS:
            #quad = list(reversed(quad))
            normal = self.totallyNormal( points[ quad[ 0 ] -1 ], points[ quad[ 1 ] -1 ], points[ quad[ 2 ] -1 ] )
            lines.append( "vn " + " ".join([str(value) for value in normal]) )

        lines.append( 'usemtl MyMaterial' )

        quadIndex = 0
        for quad in QUADS:
            quadIndex = 1 + quadIndex
            face_uv = "f " + " ".join([f'{vertex}/{vertex}/{quadIndex}' for vertex in quad])
            face_un = "f " + " ".join([str(vertex) for vertex in quad])
            if self.uvwrap:
                lines.append( face_uv )
            else:
                lines.append( f'#{face_uv}' )
                lines.append( f'{face_un}' )
                #"f " + " ".join([str(vertex) for vertex in quad]) )

        out = open( obj_file.name, 'w' )
        out.write( '\n'.join( lines ) + '\n' )
        out.close()

        ############################################

        lines = []
        lines.append( 'newmtl MyMaterial' )
        lines.append( f'map_Kd file={png_file.name}' ) # the 'file=' is a gradio hack

        out = open( mtl_file.name, 'w' )
        out.write( '\n'.join( lines ) + '\n' )
        out.close()

        ############################################

        cv2.imwrite(png_file.name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        ############################################

        print( f'I know it is special to you so I saved it to {obj_file.name} since we are friends' )
        return (obj_file.name,mtl_file.name,png_file.name)

    def landmarksToPoints( self, image:np.ndarray, width: int, height: int, ratio: float, landmark_list: landmark_pb2.NormalizedLandmarkList, depth: np.ndarray, zoe_scale: float, flat_scale: float ):
        points      = [] # 3d vertices
        coordinates = [] # 2d texture coordinates
        colors      = [] # 3d rgb info

        mins = [+np.inf] * 3
        maxs = [-np.inf] * 3

        mp_scale = 1 - zoe_scale
        print( f'zoe_scale:{zoe_scale}, mp_scale:{mp_scale}' )

        for idx, landmark in enumerate(landmark_list.landmark):
            x, y = _normalized_to_pixel_coordinates(landmark.x,landmark.y,width,height)
            color = image[y,x]
            colors.append( [value / 255 for value in color ] )
            coordinates.append( [x/width,1-y/height] )

            if depth is not None:
                landmark.z = depth[y, x] * zoe_scale + mp_scale * landmark.z

            landmark.z = landmark.z * flat_scale

            point = [landmark.x * ratio, landmark.y, landmark.z];
            for pidx,value in enumerate( point ):
                mins[pidx] = min(mins[pidx],value)
                maxs[pidx] = max(maxs[pidx],value)
            points.append( point )

        mids = [(min_val + max_val) / 2 for min_val, max_val in zip(mins, maxs)]
        for idx,point in enumerate( points ):
            points[idx] = [(val-mid) for val, mid in zip(point,mids)]

        print( f'mins: {mins}' )
        print( f'mids: {mids}' )
        print( f'maxs: {maxs}' )
        return (points,coordinates,colors)


    def totallyNormal(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        return normal.tolist()    


    def header(self):
        return ("""
                # Image to Quad Mesh

                Uses MediaPipe to detect a face in an image and convert it to a quad mesh.
                Saves to OBJ since gltf does not support quad faces.  The 3d viewer has Y pointing the opposite direction from Blender, so ya hafta spin it.

                The face depth with Zoe can be a bit much and without it is a bit generic. In blender you can fix this just by snapping to the high poly model. For photos turning it down to .4 helps, but may still need cleanup...

                Highly recommend running it locally. The 3D model has uv values in the faces, but you will have to either use the script or do some manually tomfoolery.

                Quick import result in examples/converted/movie-gallery.mp4 under files
        """)


    def footer(self):
        return ( """
            # Using the Textured Mesh in Blender
            
            There a couple of annoying steps atm after you download the obj from the 3d viewer. 
            
            You can use the script meshin-around.sh in the files section to do the conversion or manually:
            
            1. edit the file and change the mtllib line to use fun.mtl
            2. replace / delete all lines that start with 'f', eg :%s,^f.*,,
            3. uncomment all the lines that start with '#f', eg: :%s,^#f,f,
            4. save and exit
            5. create fun.mtl to point to the texture like:
            
            ```
            newmtl MyMaterial
            map_Kd fun.png
            ```
            
            Make sure the obj, mtl and png are all in the same directory
            
            Now the import will have the texture data: File -> Import -> Wavefront (obj) -> fun.obj
            
            This is all a work around for a weird hf+gradios+babylonjs bug which seems to be related to the version
            of babylonjs being used... It works fine in a local babylonjs sandbox...
            
            # Suggested Workflows
            
            Here are some workflow ideas.
            
            ## retopologize high poly face mesh
            
            1. sculpt high poly mesh in blender
            2. snapshot the face
            3. generate the mesh using the mediapipe stuff
            4. import the low poly mediapipe face
            5. snap the mesh to the high poly model
            6. model the rest of the low poly model
            7. bake the normal / etc maps to the low poly face model
            8. it's just that easy 😛
            
            Ideally it would be a plugin...
            
            ## stable diffusion integration
            
            1. generate a face in sd
            2. generate the mesh
            3. repose it and use it for further generation
            
            May need to expanded the generated mesh to cover more, maybe with
            <a href="https://github.com/shunsukesaito/PIFu" target="_blank">PIFu model</a>.
            
        """)


    def css(self):
       return ("""
            #mesh-display-output {
                max-height: 44vh;
                max-width:  44vh;
                width:auto;
                height:auto
                }
            #img-display-output {
                max-height: 28vh;
                max-width:  28vh;
                width:auto;
                height:auto
                }
        """)


face_image_to_face_mesh().demo()

# EOF
########################################################################################
