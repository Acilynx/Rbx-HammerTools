--!nonstrict
-- MapImporter (ModuleScript)
--
-- Builds a .htmp map package (written by tools/vmf2rbx) into the place:
-- parts, per-face textures, props (MeshParts), texture lights, point/spot lights and the spawn.
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
	indoorLighting: boolean?,   -- set up Lighting (default true): the map's sun if it has one, else no sun or sky light, dim ambient
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
	local magic = buffer.readstring(data, 0, 4)
	assert(magic == "HTMZ" or magic == "HTMP", "not a map package (.htmp)")
	local jlen = buffer.readu32(data, 4)
	local json
	if magic == "HTMZ" then -- the JSON part is zstd-compressed
		local packed = buffer.create(jlen)
		buffer.copy(packed, 0, data, 8, jlen)
		json = buffer.tostring(EncodingService:DecompressBuffer(packed, Enum.CompressionAlgorithm.Zstd))
	else
		json = buffer.readstring(data, 8, jlen)
	end
	local meta = HttpService:JSONDecode(json)
	assert(meta.version == 1 or meta.version == 2, "unsupported package version " .. tostring(meta.version))
	return meta, data, 8 + jlen
end

-- An image's pixels: RGBA for colour images (palettized "palz" or full "rgbaz"), raw W*H*ch
-- bytes for data maps ("zstd": ch 1 = roughness/metalness, ch 2 = normal map right/down).
local function decodeImage(data: buffer, blob0: number, img): buffer
	local W, H = img.w, img.h
	local src, palOff = data, blob0 + img.off
	if img.enc == "zstd" or img.enc == "palz" or img.enc == "rgbaz" then
		local packed = buffer.create(img.len)
		buffer.copy(packed, 0, data, blob0 + img.off, img.len)
		local raw = EncodingService:DecompressBuffer(packed, Enum.CompressionAlgorithm.Zstd)
		if img.enc ~= "palz" then
			return raw     -- data map, or full RGBA colour
		end
		src, palOff = raw, 0
	end
	local idxOff = palOff + img.ncol * 4
	local out = buffer.create(W * H * 4)
	for i = 0, W * H - 1 do
		buffer.writeu32(out, i * 4, buffer.readu32(src, palOff + buffer.readu8(src, idxOff + i) * 4))
	end
	return out
end

local function readBlob(data: buffer, blob0: number, b): buffer
	local packed = buffer.create(b.len)
	buffer.copy(packed, 0, data, blob0 + b.off, b.len)
	return EncodingService:DecompressBuffer(packed, Enum.CompressionAlgorithm.Zstd)
end

-- A prop mesh blob (see tools/vmf2rbx/props.py) -> EditableMesh. Positions are part-local
-- (already centred), triangles counter-clockwise from outside, uv v down the image.
local function buildMesh(raw: buffer, n: number, m: number): EditableMesh?
	local ok, em = pcall(function()
		return AssetService:CreateEditableMesh()
	end)
	if not ok or not em then
		return nil
	end
	local vids, nids, uids = table.create(n), table.create(n), table.create(n)
	local nOff, uOff, tOff = n * 12, n * 24, n * 32
	for i = 0, n - 1 do
		vids[i + 1] = em:AddVertex(Vector3.new(buffer.readf32(raw, i * 12), buffer.readf32(raw, i * 12 + 4), buffer.readf32(raw, i * 12 + 8)))
		nids[i + 1] = em:AddNormal(Vector3.new(buffer.readf32(raw, nOff + i * 12), buffer.readf32(raw, nOff + i * 12 + 4), buffer.readf32(raw, nOff + i * 12 + 8)))
		uids[i + 1] = em:AddUV(Vector2.new(buffer.readf32(raw, uOff + i * 8), buffer.readf32(raw, uOff + i * 8 + 4)))
	end
	for t = 0, m - 1 do
		local a = buffer.readu32(raw, tOff + t * 12) + 1
		local b = buffer.readu32(raw, tOff + t * 12 + 4) + 1
		local c = buffer.readu32(raw, tOff + t * 12 + 8) + 1
		local f = em:AddTriangle(vids[a], vids[b], vids[c])
		em:SetFaceNormals(f, { nids[a], nids[b], nids[c] })
		em:SetFaceUVs(f, { uids[a], uids[b], uids[c] })
	end
	return em
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
-- Roblox reads normal maps as R = image right, G = image up, B = out of the surface, on every
-- face except Bottom (Part and WedgePart alike), where both directions are reversed relative
-- to the colour image (measured in Studio with lit test cubes). `bottom` compensates.
local function orient(base: buffer, W: number, H: number, swap: boolean, fx: boolean, fy: boolean, kind: string?, bottom: boolean?)
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
				if bottom then r, d = -r, -d end
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
--
-- Textures pick up a new image when the property is set. A SurfaceAppearance doesn't: it only
-- reads its maps when it's parented (setting them on a parented one draws nothing, or drops
-- maps), so each SurfaceAppearance is replaced by a clone carrying the new images.
local MAP_PROPS = { "ColorMapContent", "NormalMapContent", "RoughnessMapContent", "MetalnessMapContent", "EmissiveMaskContent" }

function MapImporter.refreshImages(root: Instance): number
	local textureUsers: { [EditableImage]: { { any } } } = {}
	local appearances: { SurfaceAppearance } = {}
	local all: { [EditableImage]: boolean } = {}
	for _, t in ipairs(root:GetDescendants()) do
		local isTex, isSA = t:IsA("Texture"), t:IsA("SurfaceAppearance")
		if isTex or isSA then
			if isSA then
				table.insert(appearances, t)
			end
			for _, prop in ipairs(MAP_PROPS) do
				local img = (t :: any)[prop].Object
				if img and img:IsA("EditableImage") then
					all[img] = true
					if isTex then
						textureUsers[img] = textureUsers[img] or {}
						table.insert(textureUsers[img], { t, prop })
					end
				end
			end
		end
	end
	local remap: { [EditableImage]: EditableImage } = {}
	local made = 0
	for old in pairs(all) do
		local sz = old.Size
		local ok, img = pcall(function()
			return AssetService:CreateEditableImage({ Size = sz })
		end)
		if ok and img then
			img:WritePixelsBuffer(Vector2.zero, sz, old:ReadPixelsBuffer(Vector2.zero, sz))
			remap[old] = img
			for _, u in ipairs(textureUsers[old] or {}) do
				u[1][u[2]] = Content.fromObject(img)
			end
			made += 1
		end
	end
	for _, sa in ipairs(appearances) do
		local parent = sa.Parent
		local copy = sa:Clone()
		for _, prop in ipairs(MAP_PROPS) do
			local img = (sa :: any)[prop].Object
			if img and remap[img] then
				(copy :: any)[prop] = Content.fromObject(remap[img])
			end
		end
		sa:Destroy()
		copy.Parent = parent
	end
	for old in pairs(remap) do
		old:Destroy()
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

-- Maps with a light_environment (sunlight through windows or broken roofs): Roblox's sun
-- takes Source's direction, colour and strength; OutdoorAmbient (which Roblox applies only
-- where the sky is visible, like Source's sky ambient) takes the sky ambient; indoor ambient
-- stays dim and the sky-probe lighting stays off so it doesn't leak into closed rooms.
-- Roblox derives the sun direction from ClockTime and GeographicLatitude, so those are found
-- by search against Lighting:GetSunDirection() (no formula to get wrong).
local SUN_BRIGHTNESS_PER_INTENSITY = 8      -- set by eye
local SKY_AMBIENT_SCALE = 0.35

function MapImporter.setupSunLighting(sun)
	local L = game:GetService("Lighting")
	local want = Vector3.new(sun.direction[1], sun.direction[2], sun.direction[3]).Unit
	local best, bt, bl = -2, L.ClockTime, L.GeographicLatitude
	local function try(t: number, lat: number)
		L.ClockTime, L.GeographicLatitude = t % 24, math.clamp(lat, -90, 90)
		local d = L:GetSunDirection():Dot(want)
		if d > best then best, bt, bl = d, t % 24, math.clamp(lat, -90, 90) end
	end
	for t = 0, 23.5, 0.5 do
		for lat = -90, 90, 10 do try(t, lat) end
	end
	local st, sl = 0.25, 5
	for _ = 1, 12 do
		local ct, cl = bt, bl
		for dt = -1, 1 do
			for dl = -1, 1 do try(ct + dt * st, cl + dl * sl) end
		end
		st, sl = st / 2, sl / 2
	end
	L.ClockTime, L.GeographicLatitude = bt, bl
	L.Brightness = math.clamp(sun.intensity * SUN_BRIGHTNESS_PER_INTENSITY, 0.5, 6)
	L.ColorShift_Top = Color3.fromRGB(sun.color[1], sun.color[2], sun.color[3])
	local a = math.clamp(sun.ambient * SKY_AMBIENT_SCALE, 0, 1)
	local ac = sun.ambient_color
	L.OutdoorAmbient = Color3.fromRGB(ac[1] * a, ac[2] * a, ac[3] * a)
	L.Ambient = Color3.fromRGB(50, 52, 56)
	L.EnvironmentDiffuseScale = 0
	L.EnvironmentSpecularScale = 0
	L.GlobalShadows = true
	return math.deg(math.acos(math.clamp(best, -1, 1)))     -- remaining direction error, degrees
end

-- ------------------------------------------------------------------ build

-- ------------------------------------------------------------------ base64
--
-- A package embedded in the place for playtests is stored as text in StringValues (at most
-- 200,000 characters each). StringValues are meant for text, so the bytes go in as base64
-- (4 characters per 3 bytes) rather than raw.

local B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
local B64_DEC = table.create(256, 0)
for i = 1, 64 do B64_DEC[string.byte(B64, i) + 1] = i - 1 end

function MapImporter.base64Encode(data: buffer): string
	local n = buffer.len(data)
	local out = buffer.create(math.ceil(n / 3) * 4)
	local o = 0
	for i = 0, n - 1, 3 do
		local b0 = buffer.readu8(data, i)
		local b1 = if i + 1 < n then buffer.readu8(data, i + 1) else 0
		local b2 = if i + 2 < n then buffer.readu8(data, i + 2) else 0
		local v = b0 * 65536 + b1 * 256 + b2
		buffer.writeu8(out, o, string.byte(B64, (v // 262144) % 64 + 1))
		buffer.writeu8(out, o + 1, string.byte(B64, (v // 4096) % 64 + 1))
		buffer.writeu8(out, o + 2, if i + 1 < n then string.byte(B64, (v // 64) % 64 + 1) else 61) -- "="
		buffer.writeu8(out, o + 3, if i + 2 < n then string.byte(B64, v % 64 + 1) else 61)
		o += 4
	end
	return buffer.tostring(out)
end

function MapImporter.base64Decode(text: string): buffer
	local src = buffer.fromstring(text)
	local n = buffer.len(src)
	local pad = (if n > 0 and buffer.readu8(src, n - 1) == 61 then 1 else 0)
		+ (if n > 1 and buffer.readu8(src, n - 2) == 61 then 1 else 0)
	local out = buffer.create(n // 4 * 3 - pad)
	local outLen = buffer.len(out)
	local o = 0
	for i = 0, n - 4, 4 do
		local v = B64_DEC[buffer.readu8(src, i) + 1] * 262144 + B64_DEC[buffer.readu8(src, i + 1) + 1] * 4096
			+ B64_DEC[buffer.readu8(src, i + 2) + 1] * 64 + B64_DEC[buffer.readu8(src, i + 3) + 1]
		buffer.writeu8(out, o, v // 65536)
		if o + 1 < outLen then buffer.writeu8(out, o + 1, (v // 256) % 256) end
		if o + 2 < outLen then buffer.writeu8(out, o + 2, v % 256) end
		o += 3
	end
	return out
end

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
	local function variantFor(name: string, swap: boolean, fx: boolean, fy: boolean, bottom: boolean?): EditableImage?
		local im = imageMeta[name]
		if not im then return nil end
		local kind = if im.enc ~= "zstd" then "color" elseif im.ch == 2 then "normal" else "gray"
		bottom = bottom == true and kind == "normal"
		-- keyed by the image's data, so materials sharing one texture share its EditableImages
		local id = tostring(im.off)
		local key = id .. "|" .. tostring(swap) .. tostring(fx) .. tostring(fy) .. (if bottom then "|b" else "")
		if variants[key] == nil then
			bases[id] = bases[id] or decodeImage(data, blob0, im)
			variants[key] = orient(bases[id], im.w, im.h, swap, fx, fy, kind, bottom) or false
		end
		return variants[key] or nil
	end

	-- PBR maps per face: <image key>#n / #r / #m from the package (Portal RTX), else flat values
	-- from the material. Without any, textures get a flat full roughness (matte): Source's
	-- lightmapped surfaces have no specular unless the material asks for an envmap, and a
	-- part's default roughness makes every texture look polished.
	local function pbrFor(key: string, mat: string, swap: boolean, fx: boolean, fy: boolean, face: string?)
		local consts = (meta.materials[mat] or {}).pbr or {}
		local n = variantFor(key .. "#n", swap, fx, fy, face == "Bottom")
		local r = variantFor(key .. "#r", swap, fx, fy)
			or (if consts.r then flatImage(consts.r) elseif opts.matte ~= false then flatImage(255) else nil)
		local m = variantFor(key .. "#m", swap, fx, fy) or (if consts.m then flatImage(consts.m) else nil)
		-- glow: Portal RTX emissive mask or Source $selfillum mask
		local e = variantFor(key .. "#e", swap, fx, fy) or (if consts.e then flatImage(consts.e) else nil)
		return n, r, m, e, consts.es or 1
	end

	local byBrush: { [number]: { BasePart } } = {}
	local brushModels: { [number]: Model } = {}
	local counts = { parts = 0, textures = 0, placeholders = 0, lights = 0, props = 0, meshes = 0 }

	local meshRecs = {}
	for i, rec in ipairs(meta.parts) do
		if rec.cls == "Mesh" then
			table.insert(meshRecs, rec)      -- built with the props below (they share the mesh code)
			continue
		end
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
			if rec.kind == "overlay" then
				-- info_overlay: a paper-thin part just off the surface; only its texture shows
				p.Transparency = 1
				p.CanCollide, p.CanQuery, p.CanTouch = false, false, false
				p.CastShadow = false
			end
			if m and m.glass then
				-- Refract glass (observation windows): Source draws the room behind it, tinted.
				-- See-through in the tint colour, and no shadow so lights shine through. Water
				-- (tinted by its fog colour) is murkier, and you fall into it.
				p.Color = Color3.fromRGB(m.glass[1], m.glass[2], m.glass[3])
				p.Transparency = if m.liquid then 0.3 elseif m.energy then 0.8 else (opts.glassTransparency or 0.75)
				p.CastShadow = false
				if m.liquid or m.energy then p.CanCollide, p.CanQuery = false, false end
				if m.energy then p.Material = Enum.Material.Neon
				elseif not m.liquid then p.Material = Enum.Material.Glass end   -- Roblox's refraction
			end
			if rec.owner:sub(1, 8) == "trigger_" then
				-- trigger volumes (fizzler fields etc.) don't block anything
				p.CanCollide, p.CanQuery = false, false
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
					local nImg, rImg, mImg, eImg, eStrength = pbrFor(imgKey or mat, mat, swap, fx, fy, face)
					if nImg then t.NormalMapContent = Content.fromObject(nImg) end
					if rImg then t.RoughnessMapContent = Content.fromObject(rImg) end
					if mImg then t.MetalnessMapContent = Content.fromObject(mImg) end
					if eImg then
						t.EmissiveMaskContent = Content.fromObject(eImg)
						t.EmissiveStrength = eStrength
					end
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

	-- Shadow stand-ins: MeshParts made from an EditableMesh cast no shadow (Roblox; measured), so
	-- opaque Parts hidden just inside each mesh cast it instead (tools/vmf2rbx/shadow.py). They
	-- don't collide or take raycasts.
	local function shadowPart(r, cf: CFrame?): BasePart
		local p = Instance.new(if r.cls == "WedgePart" then "WedgePart" else "Part") :: BasePart
		p.Name = "shadow"
		p.Anchored = true
		p.CanCollide, p.CanQuery, p.CanTouch = false, false, false
		p.CastShadow = true
		p.Material = Enum.Material.SmoothPlastic
		p.TopSurface, p.BottomSurface = Enum.SurfaceType.Smooth, Enum.SurfaceType.Smooth
		-- the surface's average colour: blends in where depth precision lets it show through
		p.Color = if r.col then Color3.fromRGB(r.col[1], r.col[2], r.col[3]) else Color3.fromRGB(30, 30, 30)
		p.Size = Vector3.new(r.s[1], r.s[2], r.s[3])
		local local_ = cframeOf(r.c, r.r)
		p.CFrame = if cf then cf * local_ else local_
		return p
	end
	local shadowCount = 0

	-- props: one MeshPart per (model, material), built once and cloned per placement
	local blobMeta = {}
	for _, b in ipairs(meta.blobs or {}) do blobMeta[b.name] = b end
	-- The first placement of a mesh uses the MeshPart itself; later ones clone it (clones share
	-- the EditableMesh and the images).
	local templates: { [string]: MeshPart | false } = {}
	local templateUsed: { [string]: boolean } = {}
	local function meshPart(sm, solid: boolean, fidelity: Enum.CollisionFidelity?): MeshPart?
		local key = sm.blob .. (if solid then "|s" else "|n")
		if templates[key] == nil then
			templates[key] = false
			local b = blobMeta[sm.blob]
			local em = b and buildMesh(readBlob(data, blob0, b), sm.n, sm.m)
			if em then
				local ok, mp = pcall(function()
					return AssetService:CreateMeshPartAsync(Content.fromObject(em), {
						CollisionFidelity = if not solid then Enum.CollisionFidelity.Box
							else fidelity or Enum.CollisionFidelity.Default,
					})
				end)
				if ok and mp then
					mp.Anchored = true
					mp.CanCollide, mp.CanQuery, mp.CanTouch = solid, solid, solid
					mp.Material = Enum.Material.SmoothPlastic
					mp.Color = colorOf(meta, sm.material)
					mp:SetAttribute("SourceMaterial", sm.material)
					local img = variantFor(sm.material, false, false, false)
					if img then
						local sa = Instance.new("SurfaceAppearance")
						sa.ColorMapContent = Content.fromObject(img)
						local nImg, rImg, mImg, eImg, eStrength = pbrFor(sm.material, sm.material, false, false, false)
						if nImg then sa.NormalMapContent = Content.fromObject(nImg) end
						if rImg then sa.RoughnessMapContent = Content.fromObject(rImg) end
						if mImg then sa.MetalnessMapContent = Content.fromObject(mImg) end
						if eImg then
							sa.EmissiveMaskContent = Content.fromObject(eImg)
							sa.EmissiveStrength = eStrength
						end
						local mm = meta.materials[sm.material]
						if mm and mm.texlight and opts.texlightsAsNeon ~= false then
							sa = nil :: any
						elseif mm and mm.alpha then
							sa.AlphaMode = Enum.AlphaMode.Transparency
							-- Measured in Studio: at Transparency 0 the alpha is only a cut-out (any
							-- alpha > 0 draws solid, so fences work but glass is opaque). Any value
							-- above 0 puts the part in the transparent pass, where alpha blends; only
							-- blended materials go there ($alphatest foliage stays a cut-out: it
							-- shades better, and transparent-pass sorting hid it behind glass).
							if mm.blend then mp.Transparency = 0.02 end
							mp.CastShadow = false
						end
						if sa then sa.Parent = mp end
					end
					-- glass and water (no image: Refract/Water shaders), texture lights
					local mm = meta.materials[sm.material]
					if mm and mm.nocull then
						mp.DoubleSided = true          -- $nocull (foliage cards): both sides drawn
					end
					if mm and mm.glass then
						mp.Color = Color3.fromRGB(mm.glass[1], mm.glass[2], mm.glass[3])
						mp.Transparency = if mm.liquid then 0.3 elseif mm.energy then 0.8 else (opts.glassTransparency or 0.75)
						mp.CastShadow = false
						if mm.liquid or mm.energy then mp.CanCollide, mp.CanQuery, mp.CanTouch = false, false, false end
						if mm.energy then mp.Material = Enum.Material.Neon
						elseif not mm.liquid then mp.Material = Enum.Material.Glass end   -- Roblox's refraction
					elseif mm and mm.texlight and opts.texlightsAsNeon ~= false then
						mp.Material = Enum.Material.Neon
					end
					templates[key] = mp
				end
			end
		end
		local tpl = templates[key]
		if not tpl then
			return nil
		end
		if not templateUsed[key] then
			templateUsed[key] = true
			return tpl
		end
		return tpl:Clone()
	end
	local propFolder = folder("Props")
	for i, inst in ipairs(meta.props or {}) do
		local subs = meta.models and meta.models[inst.model]
		if subs then
			local model = Instance.new("Model")
			local stem = inst.model:match("([^/|]+)|") or "prop"
			model.Name = if inst.name ~= "" then inst.name else stem
			model:SetAttribute("SourceClass", inst.cls)
			model:SetAttribute("SourceModel", (inst.model:match("^[^|]+")))
			if inst.parent ~= "" then model:SetAttribute("SourceParent", inst.parent) end
			local cf = cframeOf(inst.c, inst.r)
			for k, sm in ipairs(subs) do
				local mp = meshPart(sm, inst.solid)
				if mp then
					mp.Name = "mesh_" .. k
					mp.CFrame = cf * CFrame.new(sm.centre[1], sm.centre[2], sm.centre[3])
					local sa = mp:FindFirstChildOfClass("SurfaceAppearance")
					if sa then
						sa.Color = Color3.fromRGB(inst.color[1], inst.color[2], inst.color[3])
					end
					mp.Parent = model
				end
			end
			for _, r in ipairs((meta.model_shadows or {})[inst.model] or {}) do
				shadowPart(r, cf).Parent = model
				shadowCount += 1
			end
			model.Parent = propFolder
			counts.props += 1
		end
		if i % 20 == 0 then
			progress(string.format("props %d/%d", i, #meta.props))
			task.wait()
		end
	end

	-- brushes the translator couldn't split into parts (bevelled trims, cylinders, domes):
	-- one MeshPart per visible material, placed at the brush centre. Brushes are convex, so a
	-- Hull collision is exact; when the visible faces don't span the whole brush (nodraw
	-- sides), or there are several materials, an invisible hull of all sides collides instead.
	for i, rec in ipairs(meshRecs) do
		local model = Instance.new("Model")
		model.Name = "brush_" .. rec.id
		model:SetAttribute("SourceId", rec.id)
		model:SetAttribute("SourceMaterial", rec.mat)
		model:SetAttribute("SourceKind", rec.kind)
		local cf = CFrame.new(rec.c[1], rec.c[2], rec.c[3])
		byBrush[rec.id] = byBrush[rec.id] or {}
		-- displacements aren't convex: precise decomposition; convex brushes: one exact hull
		local fidelity = if rec.fidelity == "precise" then Enum.CollisionFidelity.PreciseConvexDecomposition
			else Enum.CollisionFidelity.Hull
		for k, sm in ipairs(rec.meshes) do
			local mp = meshPart(sm, rec.collider == nil and not rec.nocollide, fidelity)
			if mp then
				if rec.nocollide then mp.CastShadow = false end     -- overlays: decals on surfaces
				mp.Name = "mesh_" .. k
				mp.CFrame = cf
				mp:SetAttribute("SourceId", rec.id)
				mp:SetAttribute("SourceKind", rec.kind)
				mp.Parent = model
				table.insert(byBrush[rec.id], mp)
				counts.meshes += 1
			end
		end
		if rec.collider then
			local cp = meshPart({ blob = rec.collider.blob, n = rec.collider.n, m = rec.collider.m, material = "" },
				true, Enum.CollisionFidelity.Hull)
			if cp then
				local sa = cp:FindFirstChildOfClass("SurfaceAppearance")
				if sa then sa:Destroy() end
				cp.Name = "collision"
				cp.CFrame = cf
				cp.Transparency, cp.CastShadow = 1, false
				cp:SetAttribute("SourceId", rec.id)
				cp.Parent = model
			end
		end
		for _, r in ipairs(rec.shadow or {}) do
			shadowPart(r).Parent = model
			shadowCount += 1
		end
		model.Parent = folder(folderName(rec))
		if i % 50 == 0 then
			progress(string.format("brush meshes %d/%d", i, #meshRecs))
			task.wait()
		end
	end

	-- light blockers: thick opaque boxes in the void behind the map's walls (never seen; Source
	-- maps are sealed) so the sun's shadows don't leak through thin walls and brush seams.
	-- Locked, so clicking the map from outside in Studio selects the map, not them.
	local blockerCount = 0
	if meta.blockers and #meta.blockers > 0 then
		local bf = Instance.new("Folder")
		bf.Name = "LightBlockers"
		for i, r in ipairs(meta.blockers) do
			local p = Instance.new("Part")
			p.Name = "blocker"
			p.Anchored, p.Locked = true, true
			p.CanCollide, p.CanQuery, p.CanTouch = false, false, false
			p.CastShadow = true
			p.Material = Enum.Material.SmoothPlastic
			p.TopSurface, p.BottomSurface = Enum.SurfaceType.Smooth, Enum.SurfaceType.Smooth
			p.Color = Color3.fromRGB(60, 60, 60)
			p.Size = Vector3.new(r.s[1], r.s[2], r.s[3])
			p.CFrame = CFrame.new(r.c[1], r.c[2], r.c[3])
			p.Parent = bf
			blockerCount += 1
			if i % 500 == 0 then
				progress(string.format("light blockers %d/%d", i, #meta.blockers))
				task.wait()
			end
		end
		bf.Parent = root
	end

	root.Parent = parent

	if opts.indoorLighting ~= false then
		if meta.lights.sun then
			local err = MapImporter.setupSunLighting(meta.lights.sun)
			progress(string.format("sun set up (direction within %.2f degrees)", err))
		else
			MapImporter.setupIndoorLighting()
		end
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
	progress(string.format("done: %d parts, %d brush meshes, %d textures (%d images), %d props, %d lights, %d shadow stand-ins, %d light blockers, %d placeholders",
		counts.parts, counts.meshes, counts.textures, refreshed, counts.props, counts.lights, shadowCount, blockerCount, counts.placeholders))
	if imageFailures > 0 then
		warn(string.format("[MapImporter] %d images couldn't be created (EditableImage memory budget?); those maps are missing", imageFailures))
	end
	return root, counts
end

return MapImporter
