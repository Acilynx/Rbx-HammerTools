-- ImporterPlugin (plugin Script)
--
-- Standalone toolbar button for the map importer. To be folded into the Hammer Tools plugin;
-- until then: put this Script and the MapImporter ModuleScript in one Folder in Studio,
-- right-click the Folder > Save as Local Plugin.

local StudioService = game:GetService("StudioService")
local ChangeHistoryService = game:GetService("ChangeHistoryService")
local MapImporter = require(script.Parent.MapImporter)

local toolbar = plugin:CreateToolbar("Hammer Tools")
local button = toolbar:CreateButton("Import map", "Build a .htmp map package (from tools/vmf2rbx) into the place", "")
button.ClickableWhenViewportHidden = true

button.Click:Connect(function()
	local file = StudioService:PromptImportFileAsync({ "htmp" })
	if not file then
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
	button:SetActive(false)
end)
