"""VMT parsing: Valve's own materials that srctools' strict parser rejects."""
from vmf2rbx.assets import parse_vmt


def _root(text):
    return [b for b in parse_vmt(text) if b.has_children()][0]


def test_multi_token_value():
    # materials/models/props/food_can/food_can.vmt
    r = _root('"VertexLitGeneric"\r\n{\r\n\t"$basetexture"\t"models/props/food_can/food_can"\r\n'
              '\t"$envmaptint" .6 .6 .6\r\n}')
    assert r.real_name == "VertexLitGeneric"
    assert r["$basetexture"] == "models/props/food_can/food_can"


def test_missing_close_brace_and_comments():
    # materials/models/props/sphere_lit.vmt
    r = _root('"vertexlitgeneric"\r\n{\r\n\t"$baseTexture" "models\\props/sphere_lit"\r\n'
              '//\t"$basealphaenvmapmask" 1\r\n\t"$selfillum" 1\r\n\r\n')
    assert r["$basetexture"] == "models\\props/sphere_lit"      # backslash kept, not an escape
    assert r["$selfillum"] == "1"
    assert "$basealphaenvmapmask" not in [c.name for c in r]


def test_block_comment_and_nested():
    r = _root('"Modulate"\n{\n\t"$basetexture" "a/b"\n\t/*\n\t"Proxies" { "x" { "y" "1" } }\n\t*/\n'
              '\t"Proxies"\n\t{\n\t\t"Sine"\n\t\t{\n\t\t\t"resultVar" "$alpha"\n\t\t}\n\t}\n}')
    assert r["$basetexture"] == "a/b"
    assert [c.name for c in r.find_children("proxies")] == ["sine"]
