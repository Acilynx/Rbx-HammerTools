-- ImporterPlugin (plugin Script)
--
-- Toolbar buttons for the map importer. To be folded into the Hammer Tools plugin; until then:
-- put this Script and the MapImporter ModuleScript in one Folder in Studio, right-click the
-- Folder > Save as Local Plugin.

local RunService = game:GetService("RunService")
local StudioService = game:GetService("StudioService")
local ChangeHistoryService = game:GetService("ChangeHistoryService")
local MapImporter = require(script.Parent.MapImporter)

-- ------------------------------------------------------------------ playtest watcher
--
-- After a playtest, Studio stops drawing every EditableImage in the edit place (the pixels
-- survive; only the drawing is lost), so imported maps go untextured. There's no "playtest
-- ended" event for plugins, but plugins also load into the playtest's own DataModels, and
-- plugin settings are shared between those copies. So the playtest copy writes a heartbeat
-- into a setting every second, and the edit copy refreshes the textures once the heartbeat
-- goes stale.

local BEAT_KEY = "HammerTools_PlaytestBeat"
local BEAT_EVERY, STALE_AFTER = 1, 3000 -- seconds / milliseconds

local function nowMs(): number
	return DateTime.now().UnixTimestampMillis
end

if RunService:IsRunning() then
	-- this copy lives inside a playtest: no toolbar, just the heartbeat
	while true do
		plugin:SetSetting(BEAT_KEY, nowMs())
		task.wait(BEAT_EVERY)
	end
end

local function refreshTextures(reason: string)
	local map = workspace:FindFirstChild("Map")
	if not map then
		return
	end
	local rec = ChangeHistoryService:TryBeginRecording("Refresh map textures")
	local made = MapImporter.refreshImages(map)
	if rec then
		ChangeHistoryService:FinishRecording(rec, Enum.FinishRecordingOperation.Commit)
	end
	if made > 0 then
		print(string.format("[MapImporter] refreshed %d images (%s)", made, reason))
	end
end

task.spawn(function()
	local handled = plugin:GetSetting(BEAT_KEY) -- a beat left over from an earlier session
	while true do
		task.wait(BEAT_EVERY)
		local beat = plugin:GetSetting(BEAT_KEY)
		if beat and beat ~= handled and nowMs() - beat > STALE_AFTER then
			handled = beat
			refreshTextures("playtest ended")
		end
	end
end)

-- ------------------------------------------------------------------ toolbar

local toolbar = plugin:CreateToolbar("Hammer Tools")
local importButton = toolbar:CreateButton("Import map", "Build a .htmp map package (from tools/vmf2rbx) into the place", "")
importButton.ClickableWhenViewportHidden = true
local refreshButton = toolbar:CreateButton("Refresh textures", "Redraw imported map textures (if they've gone blank)", "")
refreshButton.ClickableWhenViewportHidden = true
local embedButton = toolbar:CreateButton("Embed for playtest",
	"Store a map package in the place so a playtest (Play) builds it with textures and props", "")
embedButton.ClickableWhenViewportHidden = true

importButton.Click:Connect(function()
	local file = StudioService:PromptImportFileAsync({ "htmp" })
	if not file then
		importButton:SetActive(false)
		return
	end
	local rec = ChangeHistoryService:TryBeginRecording("Import map")
	local ok, err = pcall(MapImporter.build, file:GetBinaryContents(), {
		onProgress = function(msg)
			print("[MapImporter] " .. msg)
		end,
	})
	if rec then
		ChangeHistoryService:FinishRecording(rec, if ok then Enum.FinishRecordingOperation.Commit else Enum.FinishRecordingOperation.Cancel)
	end
	if not ok then
		warn("[MapImporter] " .. tostring(err))
	end
	importButton:SetActive(false)
end)

-- ------------------------------------------------------------------ playtest embedding
--
-- Textures and prop meshes are EditableImages/EditableMeshes: they exist only in this Studio
-- session, and a playtest doesn't get them. So the playtest rebuilds the map itself. It does
-- that on the server: a client may hold only ~80 MB of editable images and ~7 editable meshes,
-- and can't set PBR maps; the server has no limits, and MapImporter's `replicate` option turns
-- each image and mesh into content that replicates to the client.
-- The package goes into ServerStorage.HammerPlaytest (base64 text in StringValues, so it never
-- travels to the client), with a copy of MapImporter, plus ServerScriptService.HammerPlaytestLoader.
-- Studio playtests only: publishing the place would put the map's (Valve) textures on Roblox.

local CHUNK = 199000 -- StringValue.Value must stay under 200,000 characters

local LOADER_SOURCE = [==[
-- HammerPlaytestLoader: made by Hammer Tools "Embed for playtest". Builds the embedded map on
-- the server when a playtest starts; its images and meshes replicate to the client.
if not game:GetService("RunService"):IsStudio() then
	return -- Studio playtests only
end
local ServerStorage = game:GetService("ServerStorage")
local holder = ServerStorage:WaitForChild("HammerPlaytest")
local MapImporter = require(holder:WaitForChild("MapImporter"))
local data = holder:WaitForChild("Data")
local count = data:GetAttribute("Count")

local t0 = os.clock()
local pieces = table.create(count)
for i = 1, count do
	pieces[i] = data:WaitForChild(string.format("%04d", i)).Value
end
local contents = buffer.tostring(MapImporter.base64Decode(table.concat(pieces)))
print(string.format("[HammerPlaytest] package %.1f MB read in %.1f s", #contents / 1e6, os.clock() - t0))

local runtime = Instance.new("Folder")
runtime.Name = "MapRuntime"
runtime.Parent = workspace
local ok, root = pcall(MapImporter.build, contents, {
	parent = runtime,
	replicate = true,        -- images/meshes become content that reaches the client
	moveSpawn = false,       -- the spawn was already moved when the map was imported
	indoorLighting = false,  -- and the lighting set up
	onProgress = function(msg) print("[HammerPlaytest] " .. msg) end,
})
if not ok then
	warn("[HammerPlaytest] " .. tostring(root))
	return
end
-- the copy saved in the place has no textures or props during a playtest: drop it
local saved = workspace:FindFirstChild("Map")
saved = saved and saved:FindFirstChild(root.Name)
if saved then saved:Destroy() end
print(string.format("[HammerPlaytest] map built in %.1f s", os.clock() - t0))
]==]

local function embedForPlaytest(contents: string)
	local ServerStorage = game:GetService("ServerStorage")
	local ServerScriptService = game:GetService("ServerScriptService")
	-- earlier versions put these in ReplicatedStorage / StarterPlayerScripts
	local legacy = {
		ServerStorage:FindFirstChild("HammerPlaytest"),
		ServerScriptService:FindFirstChild("HammerPlaytestLoader"),
		game:GetService("ReplicatedStorage"):FindFirstChild("HammerPlaytest"),
		game:GetService("StarterPlayer").StarterPlayerScripts:FindFirstChild("HammerPlaytestLoader"),
	}
	for _, x in pairs(legacy) do
		x:Destroy()
	end

	local text = MapImporter.base64Encode(buffer.fromstring(contents))
	local holder = Instance.new("Folder")
	holder.Name = "HammerPlaytest"
	local data = Instance.new("Folder")
	data.Name = "Data"
	data.Parent = holder
	local count = math.ceil(#text / CHUNK)
	for i = 1, count do
		local v = Instance.new("StringValue")
		v.Name = string.format("%04d", i)
		v.Value = string.sub(text, (i - 1) * CHUNK + 1, i * CHUNK)
		v.Parent = data
	end
	data:SetAttribute("Count", count)
	local moduleCopy = script.Parent.MapImporter:Clone()
	moduleCopy.Parent = holder
	holder.Parent = ServerStorage

	local loader = Instance.new("Script")
	loader.Name = "HammerPlaytestLoader"
	loader.Source = LOADER_SOURCE
	loader.Parent = ServerScriptService
	return count, #text
end

embedButton.Click:Connect(function()
	local file = StudioService:PromptImportFileAsync({ "htmp" })
	if not file then
		embedButton:SetActive(false)
		return
	end
	local rec = ChangeHistoryService:TryBeginRecording("Embed map for playtest")
	local ok, count, size = pcall(embedForPlaytest, file:GetBinaryContents())
	if rec then
		ChangeHistoryService:FinishRecording(rec, if ok then Enum.FinishRecordingOperation.Commit else Enum.FinishRecordingOperation.Cancel)
	end
	if ok then
		print(string.format("[MapImporter] embedded for playtest: %.1f MB of text in %d StringValues "
			.. "(ServerStorage.HammerPlaytest). Delete it and ServerScriptService.HammerPlaytestLoader to remove.",
			size / 1e6, count))
	else
		warn("[MapImporter] " .. tostring(count))
	end
	embedButton:SetActive(false)
end)

refreshButton.Click:Connect(function()
	refreshTextures("button")
	refreshButton:SetActive(false)
end)
