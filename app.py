########################################################################################
#

import gradio as gr
import cv2
import glob
import json
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
from MediaMesh import *

class FaceMeshWorkflow:
    LOG = logging.getLogger(__name__)

    IMAGE  = 'image'
    LABEL  = 'label'
    MESH   = 'mesh'
    LO     = 'lo'     
    HI     = 'hi'     
    TO_LO  = 'toLo'   
    TO_HI  = 'toHi'   
    WEIGHT = 'weight'   
    BUTTON = 'button' 

    def __init__(self):
        self.mm = mediaMesh = MediaMesh().demoSetup()

    def demo(self):
        demo = gr.Blocks()
        sources = {source:{} for source in 'mediapipe zoe midas'.split()}
        flat_inn = []
        flat_out = []
        with demo:
            gr.Markdown(self.header())

            # image input and annotated output

            with gr.Row():
                upload_image  = gr.Image(label="Input image", type="numpy", source="upload")
                flat_inn.append( upload_image )
                examples      = gr.Examples( examples=self.examples(), inputs=[upload_image] )
                detect_button = gr.Button(value="Detect Faces")
                faced_image   = self.img('faced image')
                flat_out.append( faced_image )

            # per source widget sets

            for name, source in sources.items():
                with gr.Row():
                    source[ FaceMeshWorkflow.LABEL ] = gr.Label(label=name, value=name)
                with gr.Row():
                    source[ FaceMeshWorkflow.IMAGE ] = self.img(f'{name} depth')
                    with gr.Group():
                        source[ FaceMeshWorkflow.LO     ] = gr.Label( label=f'{name}:Min', value=+33)
                        source[ FaceMeshWorkflow.HI     ] = gr.Label( label=f'{name}:Max', value=-33)
                        source[ FaceMeshWorkflow.TO_LO  ] = gr.Slider(label=f'{name}:Target Min', value=-.11, minimum=-3.3, maximum=3.3, step=0.01)
                        source[ FaceMeshWorkflow.TO_HI  ] = gr.Slider(label=f'{name}:Target Max', value=+.11, minimum=-3.3, maximum=3.3, step=0.01)
                        source[ FaceMeshWorkflow.BUTTON ] = gr.Button(value='Update Mesh')
                    source[ FaceMeshWorkflow.MESH ] = self.m3d(name)

            # the combined mesh with controls 

            weights = []
            with gr.Row():
                with gr.Row():
                    with gr.Column():
                        for name, source in sources.items():
                            source[ FaceMeshWorkflow.WEIGHT ] = gr.Slider(label=f'{name}:Source Weight', value=1, minimum=-1, maximum=1, step=0.01)
                            weights.append( source[ FaceMeshWorkflow.WEIGHT ] )
                        combine_button = gr.Button(value="Combined Meshes")
                    with gr.Column():
                        combined_mesh = self.m3d( 'combined' )
                        flat_out.append( combined_mesh )

            # setup the button clicks 

            outties = {k:True for k in [ FaceMeshWorkflow.MESH, FaceMeshWorkflow.IMAGE, FaceMeshWorkflow.LO, FaceMeshWorkflow.HI]}
            for name, source in sources.items():
                update_inputs = []
                update_outputs = [combined_mesh, source[FaceMeshWorkflow.MESH]]
                for key, control in source.items():
                    if key is FaceMeshWorkflow.BUTTON: 
                        continue
                    if key in outties:
                        flat_out.append( control )
                    else:
                        if not key is FaceMeshWorkflow.LABEL:
                            flat_inn.append( control )
                        update_inputs.append( control )
                source[FaceMeshWorkflow.BUTTON].click( fn=self.remesh, inputs=update_inputs, outputs=update_outputs )

            detect_button.click(  fn=self.detect,   inputs=flat_inn, outputs=flat_out )
            combine_button.click( fn=self.combine,  inputs=weights, outputs=[combined_mesh] )

        demo.launch()

    def detect(self, image:np.ndarray, mp_lo, mp_hi, mp_wt, zoe_lo, zoe_hi, zoe_wt, midas_lo, midas_hi, midas_wt):
        self.mm.detect(image)

        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].weight = mp_wt
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].weight = zoe_wt
        self.mm.weightMap.values[ MidasDepthSource.NAME ].weight = midas_wt

        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].toLo = mp_lo
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].toLo = zoe_lo
        self.mm.weightMap.values[ MidasDepthSource.NAME ].toLo = midas_lo

        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].toHi = mp_hi
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].toHi = zoe_hi
        self.mm.weightMap.values[ MidasDepthSource.NAME ].toHi = midas_hi

        meshes = self.mm.meshmerizing()

        z = self.mm.depthSources[0]
        m = self.mm.depthSources[1]

        ##################################################################

        annotated = self.mm.annotated
        combined_mesh = meshes[MediaMesh.COMBINED][0]

        mp_gray = self.mm.gray
        mp_lo   = str(self.mm.weightMap.values[ DepthMap.MEDIA_PIPE ].lo)
        mp_hi   = str(self.mm.weightMap.values[ DepthMap.MEDIA_PIPE ].hi)
        mp_mesh = meshes[DepthMap.MEDIA_PIPE][0]

        zoe_gray = z.gray
        zoe_lo   = str(self.mm.weightMap.values[z.name].lo)
        zoe_hi   = str(self.mm.weightMap.values[z.name].hi)
        zoe_mesh = meshes[z.name][0]

        midas_gray = m.gray
        midas_lo   = str(self.mm.weightMap.values[m.name].lo)
        midas_hi   = str(self.mm.weightMap.values[m.name].hi)
        midas_mesh = meshes[m.name][0]

        ##################################################################
        # gotta write 'em to disk for some reason

        combined_mesh = self.writeMesh( MediaMesh.COMBINED, meshes[MediaMesh.COMBINED][0] )
        mp_mesh       = self.writeMesh( DepthMap.MEDIA_PIPE, meshes[DepthMap.MEDIA_PIPE][0] )
        zoe_mesh      = self.writeMesh( z.name, meshes[z.name][0] )
        midas_mesh    = self.writeMesh( m.name, meshes[m.name][0] )

        ##################################################################
        # [image, model3d, (image, label, label, model3d), (image, label, label, model3d), (image, label, label, model3d)]

        return annotated, combined_mesh, mp_gray, mp_lo, mp_hi, mp_mesh, zoe_gray, zoe_lo, zoe_hi, zoe_mesh, midas_gray, midas_lo, midas_hi, midas_mesh

    def combine(self, mp_wt, zoe_wt, midas_wt ):
        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].weight = mp_wt
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].weight = zoe_wt
        self.mm.weightMap.values[ MidasDepthSource.NAME ].weight = midas_wt
        return self.writeMesh(MediaMesh.COMBINED, self.mm.toObj(MediaMesh.COMBINED)[0])

    def kombine(self, image:np.ndarray, mp_lo, mp_hi, mp_wt, zoe_lo, zoe_hi, zoe_wt, midas_lo, midas_hi, midas_wt):
        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].weight = mp_wt
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].weight = zoe_wt
        self.mm.weightMap.values[ MidasDepthSource.NAME ].weight = midas_wt

        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].toLo = mp_lo
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].toLo = zoe_lo
        self.mm.weightMap.values[ MidasDepthSource.NAME ].toLo = midas_lo

        self.mm.weightMap.values[ DepthMap.MEDIA_PIPE   ].toHi = mp_hi
        self.mm.weightMap.values[ ZoeDepthSource.NAME   ].toHi = zoe_hi
        self.mm.weightMap.values[ MidasDepthSource.NAME ].toHi = midas_hi

        meshes = self.mm.meshmerizing()

        z = self.mm.depthSources[0]
        m = self.mm.depthSources[1]

        ##################################################################

        annotated = self.mm.annotated
        combined_mesh = meshes[MediaMesh.COMBINED][0]

        mp_gray = self.mm.gray
        mp_lo   = str(self.mm.weightMap.values[ DepthMap.MEDIA_PIPE ].lo)
        mp_hi   = str(self.mm.weightMap.values[ DepthMap.MEDIA_PIPE ].hi)
        mp_mesh = meshes[DepthMap.MEDIA_PIPE][0]

        zoe_gray = z.gray
        zoe_lo   = str(self.mm.weightMap.values[z.name].lo)
        zoe_hi   = str(self.mm.weightMap.values[z.name].hi)
        zoe_mesh = meshes[z.name][0]

        midas_gray = m.gray
        midas_lo   = str(self.mm.weightMap.values[m.name].lo)
        midas_hi   = str(self.mm.weightMap.values[m.name].hi)
        midas_mesh = meshes[m.name][0]

        ##################################################################
        # gotta write 'em to disk for some reason

        combined_mesh = self.writeMesh( MediaMesh.COMBINED, meshes[MediaMesh.COMBINED][0] )
        mp_mesh       = self.writeMesh( DepthMap.MEDIA_PIPE, meshes[DepthMap.MEDIA_PIPE][0] )
        zoe_mesh      = self.writeMesh( z.name, meshes[z.name][0] )
        midas_mesh    = self.writeMesh( m.name, meshes[m.name][0] )

        ##################################################################
        # [image, model3d, (image, label, label, model3d), (image, label, label, model3d), (image, label, label, model3d)]

        return annotated, combined_mesh, mp_gray, mp_lo, mp_hi, mp_mesh, zoe_gray, zoe_lo, zoe_hi, zoe_mesh, midas_gray, midas_lo, midas_hi, midas_mesh

    def remesh(self, label:Dict[str,str], lo:float, hi:float, wt:float):
        name = label[ 'label' ] # hax
        FaceMeshWorkflow.LOG.info( f'remesh {name} with lo:{lo}, hi:{hi} and wt:{wt}' )

        self.mm.weightMap.values[ name ].toLo = lo
        self.mm.weightMap.values[ name ].toHi = hi
        self.mm.weightMap.values[ name ].weight = wt

        mesh     = self.writeMesh(name,               self.mm.singleSourceMesh(name)[0])
        combined = self.writeMesh(MediaMesh.COMBINED, self.mm.toObj(MediaMesh.COMBINED)[0])

        return mesh, combined
    
    def writeMesh(self, name:str, mesh:List[str])->str:
        file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False).name
        out = open( file, 'w' )
        out.write( '\n'.join( mesh ) + '\n' )
        out.close()
        return file

    def detective(self, *args):
        for arg in args:
            wat = 'TMI' if isinstance(arg, np.ndarray) else arg
            #c = '#' if hf_hack else ''
            print( f'hi there {type(arg)} ur a nice {wat} to have ' )
        return None

    def m3d(self, name:str):
        return gr.Model3D(clear_color=3*[0],  label=f"{name} mesh", elem_id='mesh-display-output')

    def img(self, name:str, src:str='upload'):
        return gr.Image(label=name,elem_id='img-display-output',source=src)

    def examples(self) -> List[str]:
        return glob.glob('examples/*png')
        return [
            'examples/blonde-00081-399357008.png',
            'examples/dude-00110-1227390728.png',
            'examples/granny-00056-1867315302.png',
            'examples/tuffie-00039-499759385.png',
            'examples/character.png',
        ]

    def header(self):
        return ("""
                # FaceMeshWorkflow

                The process goes like this:

                1. select an input images
                2. click "Detect Faces"
                3. fine tune the different depth sources
                4. fine tune the combinations of the depth sources
                5. download the obj and have fun

                The primary motivation was that all the MediaPipe faces all looked the same.
                Usually ZoeDepth is usually better, but can be extreme. Midas works sometimes :-P

                The depth analysis is a bit slow. Especially on the hf site, so I recommend running it locally.
                Since the tuning is a post-process to the analysis you can go nuts!

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

			If you forget, the .obj has notes on how to mangle it.
            
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

			An extension would be hoopy
            
            May want to expanded the generated mesh to cover more, maybe with
            <a href="https://github.com/shunsukesaito/PIFu" target="_blank">PIFu model</a>.
        """)
 

FaceMeshWorkflow().demo()

# EOF
########################################################################################
