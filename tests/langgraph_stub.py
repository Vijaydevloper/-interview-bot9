"""Test helper: stand-in for langgraph when it is not installed."""
import sys


def use_langgraph_or_stub():
    """Use the real langgraph when installed; otherwise a minimal stand-in that follows its
    documented StateGraph semantics (nodes return partial state; conditional edges use a map)."""
    try:
        import langgraph  # noqa: F401
        return "real"
    except ImportError:
        pass
    import types

    START, END = "__start__", "__end__"

    class StateGraph:
        def __init__(self, schema):
            self.nodes, self.edges, self.cond = {}, {}, {}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def add_edge(self, a, b):
            self.edges[a] = b

        def add_conditional_edges(self, a, fn, mapping):
            assert set(mapping.values()) <= set(self.nodes) | {END}, mapping  # real langgraph checks too
            self.cond[a] = (fn, mapping)

        def compile(self):
            g = self

            class Compiled:
                def invoke(self, state):
                    state = dict(state)
                    if START in g.cond:
                        fn, mapping = g.cond[START]
                        cur = mapping[fn(state)]
                    else:
                        cur = g.edges[START]
                    while cur != END:
                        state.update(g.nodes[cur](state))
                        if cur in g.cond:
                            fn, mapping = g.cond[cur]
                            cur = mapping[fn(state)]
                        else:
                            cur = g.edges[cur]
                    return state

            return Compiled()

    mod = types.ModuleType("langgraph.graph")
    mod.StateGraph, mod.START, mod.END = StateGraph, START, END
    sys.modules["langgraph"] = types.ModuleType("langgraph")
    sys.modules["langgraph.graph"] = mod
    return "stub"
