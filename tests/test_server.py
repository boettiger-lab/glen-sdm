"""The served MCP tool must expose every fit_sdm parameter."""
import inspect

import server


def test_served_fit_sdm_exposes_every_parameter():
    core = list(inspect.signature(server.fit_sdm).parameters)
    served = list(inspect.signature(server._fit_sdm_tool).parameters)
    assert served == core
