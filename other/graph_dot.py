import pydot

graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor="yellow")


# Add nodes
my_node = pydot.Node("a", label="Foo")
graph.add_node(my_node)
# Or, without using an intermediate variable:
graph.add_node(pydot.Node("b", shape="circle"))

# Add edges
my_edge = pydot.Edge("a", "b", color="red")
graph.add_edge(my_edge)
# Or, without using an intermediate variable:
graph.add_edge(pydot.Edge("b", "c", color="blue"))


graph.add_edge(pydot.Edge("b", "d", style="dotted"))


graph.add_subgraph(sgraph=pydot.Subgraph(graph_name='my_graph', label="A"))
graph.set_bgcolor("lightyellow")
graph.get_node("b")[0].set_shape("box")

graph.write_png("out.png")