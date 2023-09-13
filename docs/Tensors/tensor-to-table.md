## Tensor to table

AFAIK, there is no built-in PyTorch function that converts a tensor to an HTML table. However, you can create your own function to do this, using Python's built-in string formatting.

```python
def tensor_to_html_table(tensor):
    html = "<table>"
    
    for row in tensor:
        html += "<tr>"
        
        for item in row:
            html += "<td>{}</td>".format(item.item())
        
        html += "</tr>"
    
    html += "</table>"
    
    return html
```

<br>
You can use this function to convert a PyTorch tensor into an HTML table string. Note that this function assumes that the tensor is 2-dimensional. If the tensor has more dimensions, you'll need a more complex function to handle that.

### Implementation

```python
import torch

tensor = torch.tensor([[7, 8], [8, 10]])
html = tensor_to_html_table(tensor)

print(html)
```

<br>
