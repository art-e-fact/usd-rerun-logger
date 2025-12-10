from pxr import Kind, Sdf, Usd, UsdGeom, UsdShade

stage = Usd.Stage.CreateNew(
    "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/simpleShading.usda"
)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

modelRoot = UsdGeom.Xform.Define(stage, "/TexModel")
Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

billboard = UsdGeom.Mesh.Define(stage, "/TexModel/card")
billboard.CreatePointsAttr(
    [(-430, -145, 0), (430, -145, 0), (430, 145, 0), (-430, 145, 0)]
)
billboard.CreateFaceVertexCountsAttr([4])
billboard.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
billboard.CreateExtentAttr([(-430, -145, 0), (430, 145, 0)])
texCoords = UsdGeom.PrimvarsAPI(billboard).CreatePrimvar(
    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
)
texCoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])


material = UsdShade.Material.Define(stage, "/TexModel/boardMat")

pbrShader = UsdShade.Shader.Define(stage, "/TexModel/boardMat/PBRShader")
pbrShader.CreateIdAttr("UsdPreviewSurface")
pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

stReader = UsdShade.Shader.Define(stage, "/TexModel/boardMat/stReader")
stReader.CreateIdAttr("UsdPrimvarReader_float2")

diffuseTextureSampler = UsdShade.Shader.Define(
    stage, "/TexModel/boardMat/diffuseTexture"
)
diffuseTextureSampler.CreateIdAttr("UsdUVTexture")
diffuseTextureSampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
    "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/uv1.png"
)
diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
    stReader.ConnectableAPI(), "result"
)
diffuseTextureSampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
    diffuseTextureSampler.ConnectableAPI(), "rgb"
)

stInput = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
stInput.Set("st")

stReader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

billboard.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
UsdShade.MaterialBindingAPI(billboard).Bind(material)

stage.Save()
