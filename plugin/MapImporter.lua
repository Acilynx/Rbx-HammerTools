--!nonstrict
-- MapImporter (ModuleScript)
--
-- Builds a .htmp map package (written by tools/vmf2rbx) into the place:
-- parts, per-face textures, texture lights, point/spot lights and the spawn point.
--
-- Textures are EditableImages in this version ("preview mode"): they live only in the current
-- Studio session, don't survive saving, and are missing during playtests (the plugin redraws
-- them in the edit place when a playtest ends). Publish mode (uploaded image assets) comes later.

local AssetService = game:GetService("AssetService")
local EncodingService = game:GetService("EncodingService")
local HttpService = game:GetService("HttpService")

local MapImporter = {}

export type Options = {
	parent: Instance?,          -- default: workspace.Map
	moveSpawn: boolean?,        -- move/hide the first SpawnLocation onto info_player_start (default true)
	texlightsAsNeon: boolean?,  -- light-emitting materials become Neon in their light colour (default true)
	indoorLighting: boolean?,   -- set up Lighting for an enclosed map: no sun or sky light, dim ambient (default true)
	matte: boolean?,            -- give textures a flat full-roughness map so they don't read as glossy (default true)
	glassTransparency: number?, -- Transparency of Refract-shader glass such as observation windows (default 0.75)
	onProgress: ((string) -> ())?,
}

local CATEGORY_COLORS = {
	concrete = Color3.fromRGB(150, 150, 150), plastic = Color3.fromRGB(232, 232, 226),
	metal = Color3.fromRGB(90, 92, 96), glass = Color3.fromRGB(170, 210, 230),
	lights = Color3.fromRGB(255, 236, 170), signage = Color3.fromRGB(230, 140, 60),
}

local NORMALS: { [Enum.NormalId]: Vector3 } = {
	[Enum.NormalId.Right] = Vector3.xAxis, [Enum.NormalId.Left] = -Vector3.xAxis,
	[Enum.NormalId.Top] = Vector3.yAxis, [Enum.NormalId.Bottom] = -Vector3.yAxis,
	[Enum.NormalId.Back] = Vector3.zAxis, [Enum.NormalId.Front] = -Vector3.zAxis,
}

-- ------------------------------------------------------------------ package reading

local function readPackage(contents: string)
	local data = buffer.fromstring(contents)
	assert(buffer.readstring(data, 0, 4) == "HTMP", "not a map package (.htmp)")
	local jlen = buffer.readu32(data, 4)
	local meta = HttpService:JSONDecode(buffer.readstring(data, 8, jlen))
	assert(meta.version == 1 or meta.version == 2, "unsupported package version " .. tostring(meta.version))
	return meta, data, 8 + jlen
end

-- An image's pixels as stored: RGBA for palettized colour images, raw W*H*ch bytes for data
-- maps (ch 1 = roughness/metalness, ch 2 = normal map right/down components).
local function decodeImage(data: buffer, blob0: number, img): buffer
	local W, H = img.w, img.h
	if img.enc == "zstd" then
		local packed = buffer.create(img.len)
		buffer.copy(packed, 0, data, blob0 + img.off, img.len)
		return EncodingService:DecompressBuffer(packed, Enum.CompressionAlgorithm.Zstd)
	end
	local palOff = blob0 + img.off
	local idxOff = palOff + img.ncol * 4
	local out = buffer.create(W * H * 4)
	for i = 0, W * H - 1 do
		buffer.writeu32(out, i * 4, buffer.readu32(data, palOff + buffer.readu8(data, idxOff + i) * 4))
	end
	return out
end

-- EditableImages share a memory budget; when it runs out, the map loses that one map instead
-- of the whole import failing.
local imageFailures = 0
local function newImage(size: Vector2, px: buffer): EditableImage?
	local ok, img = pcall(function()
		return AssetService:CreateEditableImage({ Size = size })
	end)
	if not ok or not img then
		imageFailures += 1
		return nil
	end
	img:WritePixelsBuffer(Vector2.zero, size, px)
	return img
end

-- One of the 8 orientations of a W x H image: optional transpose, then optional flips.
-- kind "color": RGBA pixels move. kind "gray": 1-channel map, written as grey RGBA.
-- kind "normal": 2-channel (right, down) components; they move like pixels and also turn with
-- the image: a transpose swaps them, a flip negates that axis (same as the translator's
-- FaceBake.normal_matrix on 90-degree faces; tests/test_textures.py checks the two agree).
-- Roblox reads normal maps as R = image right, G = image up, B = out of the surface.
local function orient(base: buffer, W: number, H: number, swap: boolean, fx: boolean, fy: boolean, kind: string?)
	local Nx, Ny = W, H
	if swap then Nx, Ny = H, W end
	local out = buffer.create(Nx * Ny * 4)
	for y = 0, Ny - 1 do
		local sy = if fy then Ny - 1 - y else y
		for x = 0, Nx - 1 do
			local sx = if fx then Nx - 1 - x else x
			local bx, by = sx, sy
			if swap then bx, by = sy, sx end
			local o = (y * Nx + x) * 4
			local i = by * W + bx
			if kind == "normal" then
				local r = buffer.readu8(base, i * 2) / 127.5 - 1
				local d = buffer.readu8(base, i * 2 + 1) / 127.5 - 1
				if swap then r, d = d, r end
				if fx then r = -r end
				if fy then d = -d end
				local z = math.sqrt(math.max(0, 1 - r * r - d * d))
				buffer.writeu8(out, o, math.clamp(math.round((r + 1) * 127.5), 0, 255))
				buffer.writeu8(out, o + 1, math.clamp(math.round((1 - d) * 127.5), 0, 255))
				buffer.writeu8(out, o + 2, math.clamp(math.round((z + 1) * 127.5), 0, 255))
				buffer.writeu8(out, o + 3, 255)
			elseif kind == "gray" then
				local g = buffer.readu8(base, i)
				buffer.writeu32(out, o, 0xFF000000 + g * 0x10101)
			else
				buffer.writeu32(out, o, buffer.readu32(base, i * 4))
			end
		end
	end
	return newImage(Vector2.new(Nx, Ny), out)
end

local flatCache: { [number]: EditableImage } = {}
-- A 4x4 single-value grey map (flat roughness or metalness), shared by every texture using it.
local function flatImage(v: number): EditableImage?
	if not flatCache[v] then
		local px = buffer.create(4 * 4 * 4)
		for i = 0, 15 do buffer.writeu32(px, i * 4, 0xFF000000 + v * 0x10101) end
		flatCache[v] = newImage(Vector2.new(4, 4), px) :: any
	end
	return flatCache[v]
end

-- ------------------------------------------------------------------ helpers

local function colorOf(meta, mat: string): Color3
	local m = meta.materials[mat]
	if m and m.texlight then
		local c = m.texlight.color
		return Color3.new(c[1], c[2], c[3])
	end
	if m and m.color then
		return Color3.fromRGB(m.color[1], m.color[2], m.color[3])
	end
	return CATEGORY_COLORS[mat:match("^([^/]+)/") or ""] or Color3.fromRGB(200, 120, 200)
end

local function cframeOf(c, r): CFrame
	return CFrame.fromMatrix(Vector3.new(c[1], c[2], c[3]),
		Vector3.new(r[1], r[2], r[3]), Vector3.new(r[4], r[5], r[6]), Vector3.new(r[7], r[8], r[9]))
end

local function folderName(rec): string
	if rec.owner == "world" then return "World" end
	if rec.owner == "func_detail" then return "Detail" end
	return rec.owner .. (if rec.entity ~= "" then "_" .. rec.entity else "")
end

-- Studio sometimes stops drawing EditableImages: ones created while a long build was
-- yielding, and all of them after a playtest (the pixels are still there; only the drawing is
-- lost). Rebuilding each image once makes them draw again. The plugin calls this after the
-- build and whenever a playtest ends.
local MAP_PROPS = { "ColorMapContent", "NormalMapContent", "RoughnessMapContent", "MetalnessMapContent" }

function MapImporter.refreshImages(root: Instance): number
	-- group the textures by image first, then swap one image at a time, so at most one extra
	-- image exists at any moment (EditableImages share a memory budget)
	local users: { [EditableImage]: { { any } } } = {}
	for _, t in ipairs(root:GetDescendants()) do
		if t:IsA("Texture") then
			for _, prop in ipairs(MAP_PROPS) do
				local img = (t :: any)[prop].Object
				if img and img:IsA("EditableImage") then
					users[img] = users[img] or {}
					table.insert(users[img], { t, prop })
				end
			end
		end
	end
	local made = 0
	for old, list in pairs(users) do
		local sz = old.Size
		local ok, img = pcall(function()
			return AssetService:CreateEditableImage({ Size = sz })
		end)
		if ok and img then
			img:WritePixelsBuffer(Vector2.zero, sz, old:ReadPixelsBuffer(Vector2.zero, sz))
			for _, u in ipairs(list) do
				u[1][u[2]] = Content.fromObject(img)
			end
			old:Destroy()
			made += 1
		end
	end
	return made
end

-- Source maps are lit only by their own lights (plus baked bounce). Roblox's default sun,
-- outdoor ambient and sky-based environment light reach inside closed rooms and wash
-- everything out, so for an enclosed map they are turned off and a dim ambient stands in for
-- bounce light. (Undo restores the previous values, since this runs inside the import's
-- change-history recording.) A place that's already set up for indoors (no sun, no outdoor
-- ambient) is left alone, so your own tuning survives re-imports.
function MapImporter.setupIndoorLighting()
	local L = game:GetService("Lighting")
	if L.Brightness == 0 and L.OutdoorAmbient == Color3.new(0, 0, 0) then
		return false
	end
	L.Brightness = 0
	L.OutdoorAmbient = Color3.new(0, 0, 0)
	L.Ambient = Color3.fromRGB(50, 52, 56)
	L.EnvironmentDiffuseScale = 0
	L.EnvironmentSpecularScale = 0
	for _, e in ipairs(L:GetChildren()) do
		if e:IsA("SunRaysEffect") then e.Enabled = false end
	end
	return true
end

-- ------------------------------------------------------------------ build

function MapImporter.build(contents: string, options: Options?)
	local opts = options or {}
	local progress = opts.onProgress or function(_: string) end
	local meta, data, blob0 = readPackage(contents)

	local parent = opts.parent
	if not parent then
		local f = workspace:FindFirstChild("Map")
		if not f then
			f = Instance.new("Folder"); f.Name = "Map"; f.Parent = workspace
		end
		parent = f
	end
	local mapName = if meta.map ~= "" then meta.map else "ImportedMap"
	local old = (parent :: Instance):FindFirstChild(mapName)
	if old then old:Destroy() end
	local root = Instance.new("Model")
	root.Name = mapName

	local folders: { [string]: Instance } = {}
	local function folder(name: string): Instance
		if not folders[name] then
			local f = Instance.new("Folder"); f.Name = name; f.Parent = root
			folders[name] = f
		end
		return folders[name]
	end

	-- images: decode lazily, one base buffer per material, one EditableImage per orientation
	local imageMeta = {}
	for _, im in ipairs(meta.images) do imageMeta[im.name] = im end
	local bases: { [string]: buffer } = {}
	local variants: { [string]: EditableImage | false } = {}
	imageFailures = 0
	table.clear(flatCache)
	local function variantFor(name: string, swap: boolean, fx: boolean, fy: boolean): EditableImage?
		local im = imageMeta[name]
		if not im then return nil end
		local key = name .. "|" .. tostring(swap) .. tostring(fx) .. tostring(fy)
		if variants[key] == nil then
			bases[name] = bases[name] or decodeImage(data, blob0, im)
			local kind = if im.enc ~= "zstd" then "color" elseif im.ch == 2 then "normal" else "gray"
			variants[key] = orient(bases[name], im.w, im.h, swap, fx, fy, kind) or false
		end
		return variants[key] or nil
	end

	-- PBR maps per face: <image key>#n / #r / #m from the package (Portal RTX), else flat values
	-- from the material. Without any, textures get a flat full roughness (matte): Source's
	-- lightmapped surfaces have no specular unless the material asks for an envmap, and a
	-- part's default roughness makes every texture look polished.
	local function pbrFor(key: string, mat: string, swap: boolean, fx: boolean, fy: boolean)
		local consts = (meta.materials[mat] or {}).pbr or {}
		local n = variantFor(key .. "#n", swap, fx, fy)
		local r = variantFor(key .. "#r", swap, fx, fy)
			or (if consts.r then flatImage(consts.r) elseif opts.matte ~= false then flatImage(255) else nil)
		local m = variantFor(key .. "#m", swap, fx, fy) or (if consts.m then flatImage(consts.m) else nil)
		return n, r, m
	end

	local byBrush: { [number]: { BasePart } } = {}
	local brushModels: { [number]: Model } = {}
	local counts = { parts = 0, textures = 0, placeholders = 0, lights = 0 }

	for i, rec in ipairs(meta.parts) do
		local placeholder = rec.cls == "Placeholder"
		local p = Instance.new(if placeholder then "Part" else rec.cls) :: BasePart
		p.Anchored = true
		p.TopSurface = Enum.SurfaceType.Smooth
		p.BottomSurface = Enum.SurfaceType.Smooth
		p.Size = Vector3.new(rec.s[1], rec.s[2], rec.s[3])
		p.CFrame = cframeOf(rec.c, rec.r)
		p.Material = Enum.Material.SmoothPlastic
		p:SetAttribute("SourceId", rec.id)
		p:SetAttribute("SourceMaterial", rec.mat)
		p:SetAttribute("SourceKind", rec.kind)
		local target: Instance
		if placeholder then
			p.Name = "unconverted_" .. rec.id
			p.Color = Color3.fromRGB(255, 60, 60)
			p.Transparency = 0.5
			p.CanCollide = false
			p.CanQuery = false   -- the movement controller treats anything queryable as solid
			target = folder("Unconverted")
			counts.placeholders += 1
		else
			p.Color = colorOf(meta, rec.mat)
			local m = meta.materials[rec.mat]
			if m and m.texlight and opts.texlightsAsNeon ~= false then
				p.Material = Enum.Material.Neon
			end
			if m and m.glass then
				-- Refract glass (observation windows): Source draws the room behind it, tinted.
				-- See-through in the tint colour, and no shadow so lights shine through.
				p.Color = Color3.fromRGB(m.glass[1], m.glass[2], m.glass[3])
				p.Transparency = opts.glassTransparency or 0.75
				p.CastShadow = false
			end
			target = folder(folderName(rec))
			if rec.piece > 0 then
				local model = brushModels[rec.id]
				if not model then
					model = Instance.new("Model")
					model.Name = "brush_" .. rec.id
					model:SetAttribute("SourceId", rec.id)
					model:SetAttribute("SourceMaterial", rec.mat)
					model.Parent = target
					brushModels[rec.id] = model
				end
				p.Name = "piece_" .. rec.piece
				target = model
			else
				p.Name = "brush_" .. rec.id
			end
			for _, f in ipairs(rec.faces) do
				-- f[10], when present, names a baked image (a face at an angle Roblox tiling can't do)
				local face, mat, swap, fx, fy, su, sv, ou, ov, imgKey = table.unpack(f)
				local img = variantFor(imgKey or mat, swap, fx, fy)
				if img then
					local t = Instance.new("Texture")
					t.Name = "SrcTex_" .. face
					t.Face = (Enum.NormalId :: any)[face]
					t.StudsPerTileU, t.StudsPerTileV = su, sv
					t.OffsetStudsU, t.OffsetStudsV = ou, ov
					t.ColorMapContent = Content.fromObject(img)
					local nImg, rImg, mImg = pbrFor(imgKey or mat, mat, swap, fx, fy)
					if nImg then t.NormalMapContent = Content.fromObject(nImg) end
					if rImg then t.RoughnessMapContent = Content.fromObject(rImg) end
					if mImg then t.MetalnessMapContent = Content.fromObject(mImg) end
					t.Parent = p
					counts.textures += 1
					local mm = meta.materials[mat]
					if mm and mm.alpha then
						p.Transparency = 1     -- see-through materials: the texture's alpha decides
						p.CastShadow = false
					end
				end
			end
			counts.parts += 1
		end
		p.Parent = target
		byBrush[rec.id] = byBrush[rec.id] or {}
		table.insert(byBrush[rec.id], p)
		if i % 200 == 0 then
			progress(string.format("parts %d/%d", i, #meta.parts))
			task.wait()
		end
	end

	-- texture lights: a SurfaceLight on the part face that matches the emitting side
	for _, s in ipairs(meta.lights.surface) do
		local n = Vector3.new(s.normal[1], s.normal[2], s.normal[3])
		for _, p in ipairs(byBrush[s.brush] or {}) do
			if p:IsA("Part") then
				for face, ln in pairs(NORMALS) do
					if p.CFrame:VectorToWorldSpace(ln):Dot(n) > 0.99 then
						local L = Instance.new("SurfaceLight")
						L.Name = "TexLight"; L.Face = face; L.Angle = 180   -- emit over the whole hemisphere
						L.Color = Color3.fromRGB(s.color[1], s.color[2], s.color[3])
						L.Brightness = s.brightness; L.Range = s.range; L.Shadows = true
						L.Parent = p
						counts.lights += 1
					end
				end
			end
		end
	end
	local lightFolder = folder("Lights")
	for i, l in ipairs(meta.lights.points) do
		local pos = Vector3.new(l.position[1], l.position[2], l.position[3])
		local dir = Vector3.new(l.direction[1], l.direction[2], l.direction[3])
		local holder = Instance.new("Part")
		holder.Name = l.kind .. "_" .. i
		holder.Size = Vector3.new(0.2, 0.2, 0.2)
		holder.Anchored, holder.CanCollide, holder.CanQuery, holder.CanTouch = true, false, false, false
		holder.Transparency, holder.CastShadow = 1, false
		holder.CFrame = if dir.Magnitude > 0.5 then CFrame.lookAt(pos, pos + dir) else CFrame.new(pos)
		local L: Light
		if l.kind == "spot" then
			local sl = Instance.new("SpotLight")
			sl.Face = Enum.NormalId.Front
			sl.Angle = math.min(l.angle or 90, 180)
			L = sl
		else
			L = Instance.new("PointLight")
		end
		L.Color = Color3.fromRGB(l.color[1], l.color[2], l.color[3])
		L.Brightness = l.brightness
		;(L :: any).Range = l.range
		L.Shadows = true
		L.Parent = holder
		holder.Parent = lightFolder
		counts.lights += 1
	end

	root.Parent = parent

	if opts.indoorLighting ~= false then
		MapImporter.setupIndoorLighting()
	end

	if meta.spawn and opts.moveSpawn ~= false then
		local spawn = workspace:FindFirstChildOfClass("SpawnLocation")
		if spawn then
			spawn.Size = Vector3.new(4, 1, 4)
			spawn.CFrame = CFrame.new(meta.spawn[1], meta.spawn[2] - 0.5, meta.spawn[3])
			spawn.Transparency, spawn.CanCollide, spawn.CanQuery = 1, false, false
			for _, d in ipairs(spawn:GetChildren()) do
				if d:IsA("Decal") then d.Transparency = 1 end
			end
		end
	end

	progress("refreshing images")
	local refreshed = MapImporter.refreshImages(root)
	progress(string.format("done: %d parts, %d textures (%d images), %d lights, %d placeholders",
		counts.parts, counts.textures, refreshed, counts.lights, counts.placeholders))
	if imageFailures > 0 then
		warn(string.format("[MapImporter] %d images couldn't be created (EditableImage memory budget?); those maps are missing", imageFailures))
	end
	return root, counts
end

return MapImporter
