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

refreshButton.Click:Connect(function()
	refreshTextures("button")
	refreshButton:SetActive(false)
end)
