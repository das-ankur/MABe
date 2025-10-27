from torchview import draw_graph
import torch
from src.model import MABeEncoder

model = MABeEncoder(n_outputs=5)
x = torch.randn(1, 16, 232)

# Draw and save as PNG
graph = draw_graph(model, input_data=x, expand_nested=True)
graph.visual_graph.render('encoder_model_structure', format='png')
