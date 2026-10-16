"""Hammer -> Roblox map converter: a small window around `python -m vmf2rbx build`.

Start it with "Convert Map.bat" in the repo folder (or `pyw -3 tools/converter_gui.py`).
Plain Python + tkinter (included with the python.org installer); the build itself runs as a
separate process, so a crash in it can't take the window down, and its stage messages
("## ...") stream into the log.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import tkinter as tk                                   # noqa: E402
from tkinter import filedialog, messagebox, ttk        # noqa: E402

from vmf2rbx import locate                             # noqa: E402

REQUIREMENTS = os.path.join(TOOLS, "requirements.txt")
NEEDED = "import srctools, numpy, PIL, xxhash, zstandard"
CHOOSE = "Other game folder..."
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


SUPPORTED = ((3, 10), (3, 14))     # Pythons srctools/numpy ship ready-built packages for
PREFERRED = (3, 13)


def _console(exe: str) -> str:
    """python.exe rather than pythonw.exe (pythonw has no stdout to stream)."""
    d, name = os.path.split(exe)
    if name.lower() == "pythonw.exe" and os.path.isfile(os.path.join(d, "python.exe")):
        return os.path.join(d, "python.exe")
    return exe


def parse_py_list(text: str) -> list[tuple[tuple[int, int], str]]:
    """`py -0p` output -> [((major, minor), path)]. Lines look like
    ' -V:3.13 *        C:\\...\\python.exe' (newer launcher) or ' -3.13-64        C:\\...'."""
    import re
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*-(?:V:)?(\d+)\.(\d+)\S*\s+(?:\*\s+)?(.+?\.exe)\s*$", line, re.I)
        if m:
            out.append(((int(m.group(1)), int(m.group(2))), m.group(3)))
    return out


def installed_pythons() -> list[tuple[tuple[int, int], str]]:
    """This interpreter plus the ones the Windows py launcher knows about."""
    found = [(sys.version_info[:2], _console(sys.executable))]
    try:
        r = subprocess.run(["py", "-0p"], capture_output=True, text=True, creationflags=NO_WINDOW, timeout=20)
        found += [(v, _console(p)) for v, p in parse_py_list(r.stdout + r.stderr)]
    except (OSError, subprocess.SubprocessError):
        pass
    seen, out = set(), []
    for v, p in found:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen and os.path.isfile(p):
            seen.add(k)
            out.append((v, p))
    return out


def has_packages(exe: str) -> bool:
    try:
        return subprocess.run([exe, "-c", NEEDED], capture_output=True, creationflags=NO_WINDOW,
                              timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose_python(pythons, has=has_packages) -> tuple[str | None, bool]:
    """(interpreter for the build, whether it already has the packages). Prefers one that has
    them; otherwise a supported version to install into (3.13 first): a Python newer than
    srctools supports can't install it without a C compiler."""
    rank = lambda vp: (vp[0] != PREFERRED, -vp[0][1])
    for v, p in sorted(pythons, key=rank):
        if has(p):
            return p, True
    ok = [vp for vp in pythons if SUPPORTED[0] <= vp[0] <= SUPPORTED[1]]
    return (sorted(ok, key=rank)[0][1], False) if ok else (None, False)


_BUILDER: list = []


def builder_python() -> str:
    """The interpreter that runs the build (chosen once per session by App.check_packages)."""
    if os.environ.get("VMF2RBX_PYTHON"):
        return os.environ["VMF2RBX_PYTHON"]
    return _BUILDER[0] if _BUILDER else _console(sys.executable)


def settings_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "HammerTools", "converter.json")


def load_settings() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_settings(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
        with open(settings_path(), "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except OSError:
        pass


def summarize(report: dict, out_path: str) -> str:
    parts = report.get("parts", {})
    bits = [f"{parts.get('Part', 0) + parts.get('WedgePart', 0)} parts"]
    if parts.get("Mesh"):
        bits.append(f"{parts['Mesh']} brush meshes")
    props = sum(v for k, v in report.get("counts", {}).items()
                if k.startswith("props: prop_"))                 # "props: prop_static" etc.
    if props:
        bits.append(f"{props} props")
    lines = [f"Done: {os.path.basename(out_path)}  ({', '.join(bits)})"]
    if report.get("problems"):
        lines.append(f"{len(report['problems'])} brush(es) failed the self-check (see the log)")
    if parts.get("Placeholder"):
        lines.append(f"{parts['Placeholder']} brush(es) couldn't be converted (red placeholders)")
    if report.get("missing_models"):
        lines.append(f"{len(report['missing_models'])} model(s) not found in the game files")
    return "\n".join(lines)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_settings()
        self.games = locate.find_games(locate.steam_libraries(self.cfg.get("steam_roots", [])))
        self.games.update(self.cfg.get("custom_games", {}))
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.out_lines: list[str] = []

        root.title("Hammer to Roblox map converter")
        root.minsize(640, 460)
        f = ttk.Frame(root, padding=12)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="1. Decompile the map with BSPSource (it writes <map>_d.vmf).\n"
                          "2. Pick that file below and press Build.\n"
                          "3. In Studio: Import map, and pick the .htmp file.",
                  justify="left").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self.vmf = tk.StringVar(value=self.cfg.get("vmf", ""))
        self.game = tk.StringVar()
        self.bsp = tk.StringVar()
        self.out = tk.StringVar()
        self.props = tk.BooleanVar(value=self.cfg.get("props", True))
        self.verify = tk.BooleanVar(value=self.cfg.get("verify", True))

        self._row(f, 1, "Map (.vmf)", self.vmf, self.pick_vmf)
        ttk.Label(f, text="Game").grid(row=2, column=0, sticky="w", pady=3)
        self.game_box = ttk.Combobox(f, textvariable=self.game, state="readonly",
                                     values=list(self.games) + [CHOOSE])
        self.game_box.grid(row=2, column=1, sticky="ew", pady=3)
        self.game_box.bind("<<ComboboxSelected>>", self.on_game)
        self.game_path = ttk.Label(f, text="", foreground="gray")
        self.game_path.grid(row=3, column=1, columnspan=2, sticky="w")
        self._row(f, 4, "Compiled map (.bsp)", self.bsp, self.pick_bsp,
                  hint="optional: the lights come from it")
        self._row(f, 6, "Save as (.htmp)", self.out, self.pick_out)

        opts = ttk.Frame(f)
        opts.grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(opts, text="Props (models)", variable=self.props).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opts, text="Self-check every brush (slower)", variable=self.verify).pack(side="left")

        btns = ttk.Frame(f)
        btns.grid(row=9, column=0, columnspan=3, sticky="ew", pady=10)
        self.build_btn = ttk.Button(btns, text="Build", command=self.build)
        self.build_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        self.install_btn = ttk.Button(btns, text="Install required Python packages", command=self.install)
        self.open_btn = ttk.Button(btns, text="Open output folder", command=self.open_folder, state="disabled")
        self.open_btn.pack(side="right")
        self.bar = ttk.Progressbar(btns, mode="determinate", length=140)
        self.bar.pack(side="right", padx=10)

        self.status = ttk.Label(f, text="", justify="left")
        self.status.grid(row=10, column=0, columnspan=3, sticky="w")
        logf = ttk.Frame(f)
        logf.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        f.rowconfigure(11, weight=1)
        self.log = tk.Text(logf, height=12, wrap="none", font=("Consolas", 9), state="disabled")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        if self.vmf.get():
            self.on_vmf(keep=True)
        if not self.games:
            self.say("No Steam games found automatically: pick the game folder (the one with gameinfo.txt).")
        self.root.after(100, self.check_packages)
        self.root.after(100, self.pump)

    # ------------------------------------------------------------------ layout helpers
    def _row(self, f, r, label, var, cmd, hint=None):
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
        e = ttk.Entry(f, textvariable=var)
        e.grid(row=r, column=1, sticky="ew", pady=3)
        # long paths: keep the file name in view
        var.trace_add("write", lambda *_: self.root.after_idle(lambda: e.xview_moveto(1.0)))
        e.bind("<Configure>", lambda _ev: e.xview_moveto(1.0))
        ttk.Button(f, text="Browse...", command=cmd).grid(row=r, column=2, padx=(6, 0))
        if hint:
            ttk.Label(f, text=hint, foreground="gray").grid(row=r + 1, column=1, sticky="w")

    def say(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip("\n") + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ picking files
    def pick_vmf(self):
        p = filedialog.askopenfilename(title="Decompiled map", filetypes=[("Hammer map", "*.vmf"), ("All files", "*.*")],
                                       initialdir=os.path.dirname(self.vmf.get()) or None)
        if p:
            self.vmf.set(os.path.normpath(p))
            self.on_vmf()

    def on_vmf(self, keep=False):
        p = self.vmf.get()
        name, bsp = locate.guess_map(p, self.games)
        if name:
            self.game.set(name)
        elif not self.game.get() and self.cfg.get("game") in self.games:
            self.game.set(self.cfg["game"])
        self.on_game()
        self.bsp.set(bsp or "")
        if not keep or not self.out.get():
            self.out.set(locate.default_output(p))

    def on_game(self, _event=None):
        if self.game.get() == CHOOSE:
            d = filedialog.askdirectory(title="Game folder (the one with gameinfo.txt)")
            if d and os.path.isfile(os.path.join(d, "gameinfo.txt")):
                name = os.path.basename(os.path.normpath(d))
                self.games[name] = os.path.normpath(d)
                self.cfg.setdefault("custom_games", {})[name] = os.path.normpath(d)
                self.game_box.configure(values=list(self.games) + [CHOOSE])
                self.game.set(name)
            else:
                if d:
                    messagebox.showwarning("Not a game folder", "That folder has no gameinfo.txt.")
                self.game.set("")
        self.game_path.configure(text=self.games.get(self.game.get(), ""))

    def pick_bsp(self):
        start = os.path.join(self.games.get(self.game.get(), ""), "maps")
        p = filedialog.askopenfilename(title="Compiled map", filetypes=[("Compiled map", "*.bsp"), ("All files", "*.*")],
                                       initialdir=start if os.path.isdir(start) else None)
        if p:
            self.bsp.set(os.path.normpath(p))

    def pick_out(self):
        p = filedialog.asksaveasfilename(title="Save package as", defaultextension=".htmp",
                                         filetypes=[("Map package", "*.htmp")],
                                         initialfile=os.path.basename(self.out.get()) or None,
                                         initialdir=os.path.dirname(self.out.get()) or None)
        if p:
            self.out.set(os.path.normpath(p))

    # ------------------------------------------------------------------ running things
    def check_packages(self):
        if os.environ.get("VMF2RBX_PYTHON"):
            exe, ready = os.environ["VMF2RBX_PYTHON"], has_packages(os.environ["VMF2RBX_PYTHON"])
        else:
            exe, ready = choose_python(installed_pythons())
        _BUILDER[:] = [exe] if exe else []
        if exe is None:
            self.say("No usable Python found: install Python 3.13 from python.org, then reopen this window.")
            self.build_btn.configure(state="disabled")
            return
        if ready:
            self.say(f"Using {exe}")
            self.install_btn.pack_forget()
            self.build_btn.configure(state="normal")
        else:
            self.say(f"The Python packages the converter needs aren't installed yet ({exe}).")
            self.say("Press 'Install required Python packages' (needs an internet connection).")
            self.install_btn.pack(side="left", padx=6)
            self.build_btn.configure(state="disabled")

    def install(self):
        self.run([builder_python(), "-m", "pip", "install", "-r", REQUIREMENTS], "Installing packages...",
                 done=lambda ok: (self.say("Packages installed." if ok else "Install failed, see above."),
                                  self.check_packages()))

    def build(self):
        vmf, game, out = self.vmf.get(), self.games.get(self.game.get()), self.out.get()
        if not os.path.isfile(vmf):
            messagebox.showerror("No map", "Pick the decompiled .vmf file first.")
            return
        if not game:
            if not messagebox.askyesno("No game folder", "Without the game folder there are no textures or models. Build anyway?"):
                return
        cmd = [builder_python(), "-u", "-m", "vmf2rbx", "build", "--vmf", vmf, "-o", out]
        if game:
            cmd += ["--game", game]
        if os.path.isfile(self.bsp.get()):
            cmd += ["--bsp", self.bsp.get()]
        if not self.props.get():
            cmd.append("--no-props")
        if not self.verify.get():
            cmd.append("--no-verify")
        self.cfg.update(vmf=vmf, game=self.game.get(), props=self.props.get(), verify=self.verify.get())
        save_settings(self.cfg)
        self.open_btn.configure(state="disabled")
        self.run(cmd, f"Building {os.path.basename(out)}...", done=lambda ok: self.finished(ok, out))

    def run(self, cmd, title, done):
        self.out_lines = []
        self.in_report = False
        self.status.configure(text=title)
        self.say(f"> {title}")
        self.build_btn.configure(state="disabled")
        self.install_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.bar.configure(mode="indeterminate")
        self.bar.start(12)
        self.done = done

        def worker():
            try:
                self.proc = subprocess.Popen(cmd, cwd=TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
                for line in self.proc.stdout:
                    self.lines.put(("line", line))
                self.lines.put(("end", self.proc.wait() == 0))
            except Exception as ex:                  # e.g. python.exe not found
                self.lines.put(("line", f"couldn't start: {ex}\n"))
                self.lines.put(("end", False))
        threading.Thread(target=worker, daemon=True).start()

    def pump(self):
        try:
            while True:
                kind, val = self.lines.get_nowait()
                if kind == "line":
                    self.out_lines.append(val)
                    if val.startswith("## "):
                        self.status.configure(text=val[3:].strip().capitalize() + "...")
                    if val.rstrip() == "{":
                        self.in_report = True                     # the JSON report is summarized instead
                    if not self.in_report:
                        self.say("  - " + val[3:] if val.startswith("## ") else val)
                else:
                    self.bar.stop()
                    self.bar.configure(mode="determinate", value=0)
                    self.cancel_btn.configure(state="disabled")
                    self.build_btn.configure(state="normal")
                    self.install_btn.configure(state="normal")
                    self.proc = None
                    self.done(val)
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def finished(self, ok: bool, out: str):
        text = "".join(self.out_lines)
        report = None
        if "{" in text:
            try:
                report = json.loads(text[text.index("\n{") + 1:] if "\n{" in text else text[text.index("{"):])
            except ValueError:
                report = None
        if report is not None and os.path.isfile(out):
            msg = summarize(report, out)
            for p in report.get("problems", [])[:20]:
                self.say(f"  problem: {p}")
            for m in report.get("missing_models", [])[:20]:
                self.say(f"  missing model: {m}")
            self.status.configure(text=msg)
            self.say(msg)
            self.open_btn.configure(state="normal")
        else:
            self.status.configure(text="The build failed: the error is at the end of the log.")
        self.last_out = out

    def cancel(self):
        if self.proc:
            self.proc.terminate()
            self.say("Cancelled.")

    def open_folder(self):
        d = os.path.dirname(getattr(self, "last_out", "") or self.out.get())
        if sys.platform == "win32":
            os.startfile(d)                           # noqa: S606
        else:
            subprocess.Popen(["xdg-open", d])


def main():
    root = tk.Tk()
    try:
        App(root)
    except Exception:
        messagebox.showerror("Converter crashed", traceback.format_exc())
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
