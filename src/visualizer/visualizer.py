import sys
sys.path.append('../')
from ..pycore.tikzeng import *
from ..pycore.blocks  import *

redefine_arrow_style = r"""
\tikzset{connection/.style={line width=0.5mm,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7}}
"""

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    redefine_arrow_style,

    # Input Image
    to_input('../assets/sample_rgb.png', to='(2,0,0)', width=5, height=5, name='rgb_image'),
    to_input('../assets/sample_gndvi.png', to='(1,0,0)', width=5, height=5, name='gndvi_image'),
    to_input('../assets/sample_ndvi.png', to='(0,0,0)', width=5, height=5, name='ndvi_image'),

    # --- First ConvLSTM Cell ---
    to_Conv( name='convlstm1',
             s_filer="64 x 64",
             n_filer=64,
             offset="(0,0,0)",
             to="(6,0,0)",
             width=8, height=35, depth=35,
             caption="ConvLSTM Cell 1"
            ),
            to_connection(
                "rgb_image", 
                "convlstm1"
            ),

    # --- Second ConvLSTM Cell ---
    to_Conv( name='convlstm2',
             s_filer="64 x 64",
             n_filer=128,
             offset="(4,0,0)",
             to="(convlstm1-east)",
             width=16, height=35, depth=35,
             caption="ConvLSTM Cell 2"
            ),
            to_connection(
                "convlstm1", 
                "convlstm2"
            ),

    # --- Classifier Head ---
    # 1. Global Average Pooling to flatten the spatial dimensions
    to_Sum( name="gap",
            offset="(4,0,0)",
            to="(convlstm2-east)",
            radius=2.5,
           ),
           to_connection(
                "convlstm2", 
                "gap"
            ),


    # 2. Fully Connected Layer to produce logits
    to_Conv( name="fc_logits",
             s_filer="",
             n_filer="10",
             offset="(3,0,0)",
             to="(gap-east)",
             width=1,
             height=25,
             depth=1,
             caption="Logits"
          ),
          to_connection(
                "gap", 
                "fc_logits"
            ),

    to_end()
]

def main():
    to_generate(arch, 'diagram' + '.tex')

if __name__ == '__main__':
    main()
