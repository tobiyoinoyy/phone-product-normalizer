#!/usr/bin/env python3
"""一个无需命令行知识的手机商品图处理窗口（macOS/Windows/Linux）。

GUI 只是批处理入口的外壳，真正的抠图、去支架、摆正和裁切仍由
``normalize_ben2.py`` 完成。默认模型是 ``BiRefNet_dynamic``。
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

try:
    from .batch_normalize import (
        DEFAULT_MAX_GEOMETRY_PIXELS,
        DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
        DEFAULT_BIREFNET_MAX_SIDE,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT_DIR,
        BatchResult,
        collect_image_paths,
        process_files,
    )
except ImportError:  # direct ``python scripts/phone_normalizer_gui.py`` execution
    from batch_normalize import (  # type: ignore
        DEFAULT_MAX_GEOMETRY_PIXELS,
        DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE,
        DEFAULT_BIREFNET_MAX_SIDE,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT_DIR,
        BatchResult,
        collect_image_paths,
        process_files,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELP_TEXT = (
    "支持 JPG、PNG、WEBP、TIFF、BMP。\n"
    "默认使用 BiRefNet_dynamic，自动抠图、去支架、摆正并紧贴手机边缘裁切。\n"
    f"内存保护：动态模型最长边默认不超过 {DEFAULT_BIREFNET_MAX_SIDE} px；"
    f"MPS 内存不足会自动改用 CPU（上限 {DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE} px）；"
    f"几何阶段源图像上限 {DEFAULT_MAX_GEOMETRY_PIXELS / 1_000_000:.0f} MP。\n"
    "也可以直接把图片/文件夹拖到项目目录中的“手机图片一键处理.command”。"
)


class NormalizerWindow:
    """Small Tk window that keeps model work off the UI thread."""

    def __init__(
        self,
        root: Any,
        initial_paths: Optional[Sequence[Path]] = None,
        *,
        auto_start: bool = False,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk
        self.root = root
        self.root.title("手机商品图一键处理")
        self.root.geometry("820x720")
        self.root.minsize(680, 580)
        self.root.configure(bg="#f5f6f8")

        self.paths: List[Path] = []
        self.events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.running = False
        self.auto_start = bool(auto_start)
        self.cancel_event = threading.Event()
        self.started_at: Optional[float] = None
        self._was_cancelled = False
        self._close_requested = False
        self._closed = False
        self._last_total = 0
        self._completed_count = 0
        self._failed_count = 0

        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.device_var = tk.StringVar(value="auto")
        self.recursive_var = tk.BooleanVar(value=True)
        self.metadata_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="请选择图片或文件夹")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.stage_var = tk.StringVar(value="等待选择图片")
        self.count_var = tk.StringVar(value="成功 0  ·  失败 0  ·  共 0")
        self.current_var = tk.StringVar(value="当前文件：—")
        self.output_status_var = tk.StringVar(value=f"输出位置：{DEFAULT_OUTPUT_DIR}")
        self.elapsed_var = tk.StringVar(value="用时：—")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if initial_paths:
            self._add_paths(initial_paths)
            if self.auto_start and self.paths:
                # Let Tk finish drawing the window first.  This is especially
                # important when the launcher was invoked by Finder via a
                # drag-and-drop operation: the designer sees the window and
                # its initial state before model loading begins.
                self.root.after(180, self._start)
        self.root.after(120, self._poll_events)

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk

        outer = tk.Frame(self.root, bg="#f5f6f8", padx=24, pady=20)
        outer.pack(fill="both", expand=True)

        title = tk.Label(
            outer,
            text="手机商品图一键处理",
            font=("PingFang SC", 20, "bold"),
            bg="#f5f6f8",
            fg="#1f2937",
            anchor="w",
        )
        title.pack(fill="x")
        subtitle = tk.Label(
            outer,
            text="BiRefNet_dynamic · 透明 PNG · 自动去支架与角度修正",
            font=("PingFang SC", 11),
            bg="#f5f6f8",
            fg="#5b6472",
            anchor="w",
        )
        subtitle.pack(fill="x", pady=(3, 14))

        help_label = tk.Label(
            outer,
            text=HELP_TEXT,
            justify="left",
            font=("PingFang SC", 10),
            bg="#eaf2ff",
            fg="#29466d",
            padx=14,
            pady=10,
            anchor="w",
        )
        help_label.pack(fill="x", pady=(0, 14))

        source_frame = ttk.LabelFrame(outer, text=" 1. 待处理图片 ", padding=10)
        source_frame.pack(fill="both", expand=True)

        list_frame = tk.Frame(source_frame)
        list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(
            list_frame,
            selectmode="extended",
            height=8,
            font=("PingFang SC", 10),
            activestyle="none",
            relief="flat",
            bg="white",
            highlightthickness=1,
            highlightbackground="#d7dce3",
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        source_buttons = tk.Frame(source_frame)
        source_buttons.pack(fill="x", pady=(9, 0))
        self.add_files_button = ttk.Button(source_buttons, text="添加图片…", command=self._choose_files)
        self.add_files_button.pack(side="left")
        self.add_folder_button = ttk.Button(source_buttons, text="添加文件夹…", command=self._choose_folder)
        self.add_folder_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(source_buttons, text="清空", command=self._clear_paths)
        self.clear_button.pack(side="left", padx=(8, 0))

        settings = ttk.LabelFrame(outer, text=" 2. 输出设置 ", padding=10)
        settings.pack(fill="x", pady=(14, 0))

        output_row = tk.Frame(settings)
        output_row.pack(fill="x", pady=(0, 8))
        tk.Label(output_row, text="结果文件夹", width=10, anchor="w", font=("PingFang SC", 10)).pack(side="left")
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.output_button = ttk.Button(output_row, text="选择…", command=self._choose_output)
        self.output_button.pack(side="left", padx=(8, 0))

        model_row = tk.Frame(settings)
        model_row.pack(fill="x")
        tk.Label(model_row, text="抠图模型", width=10, anchor="w", font=("PingFang SC", 10)).pack(side="left")
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=("ZhengPeng7/BiRefNet_dynamic", "ZhengPeng7/BiRefNet"),
            state="readonly",
            width=31,
        )
        self.model_combo.pack(side="left")
        tk.Label(model_row, text="设备", width=7, anchor="e", font=("PingFang SC", 10)).pack(side="left", padx=(18, 0))
        self.device_combo = ttk.Combobox(
            model_row,
            textvariable=self.device_var,
            values=("auto", "mps", "cpu", "cuda"),
            state="readonly",
            width=8,
        )
        self.device_combo.pack(side="left", padx=(7, 0))
        self.recursive_check = ttk.Checkbutton(
            model_row,
            text="文件夹包含子文件夹",
            variable=self.recursive_var,
        )
        self.recursive_check.pack(side="left", padx=(18, 0))
        self.metadata_check = ttk.Checkbutton(
            model_row,
            text="保存 JSON 记录",
            variable=self.metadata_var,
        )
        self.metadata_check.pack(side="left", padx=(12, 0))

        memory_hint = ttk.Label(
            settings,
            text=(
                f"内存保护已开启：动态 BiRefNet 最长边 ≤ {DEFAULT_BIREFNET_MAX_SIDE} px；"
                f"MPS 内存不足自动切换 CPU（≤ {DEFAULT_BIREFNET_CPU_FALLBACK_MAX_SIDE} px）；"
                f"几何阶段源图像 ≤ {DEFAULT_MAX_GEOMETRY_PIXELS / 1_000_000:.0f} MP；"
                "每张图片完成后清理推理缓存。"
            ),
            justify="left",
            wraplength=730,
        )
        memory_hint.pack(fill="x", pady=(8, 0))

        # A dedicated status panel keeps the information visible while the
        # model is loading and while each image is being inferred.  The old
        # one-line status label was easy to miss and did not show successes or
        # failures until the very end.
        status_frame = ttk.LabelFrame(outer, text=" 3. 处理进度与状态 ", padding=10)
        status_frame.pack(fill="x", pady=(14, 0))

        progress_row = tk.Frame(status_frame)
        progress_row.pack(fill="x")
        self.progress = ttk.Progressbar(
            progress_row,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.progress_percent = ttk.Label(
            progress_row,
            textvariable=self.progress_percent_var,
            width=6,
            anchor="e",
        )
        self.progress_percent.pack(side="right")

        summary_row = tk.Frame(status_frame)
        summary_row.pack(fill="x", pady=(8, 0))
        self.stage_label = ttk.Label(summary_row, textvariable=self.stage_var)
        self.stage_label.pack(side="left")
        self.count_label = ttk.Label(
            summary_row,
            textvariable=self.count_var,
            anchor="e",
        )
        self.count_label.pack(side="right")

        current_label = ttk.Label(
            status_frame,
            textvariable=self.current_var,
            anchor="w",
        )
        current_label.pack(fill="x", pady=(6, 0))
        output_label = ttk.Label(
            status_frame,
            textvariable=self.output_status_var,
            anchor="w",
        )
        output_label.pack(fill="x", pady=(3, 0))

        elapsed_label = ttk.Label(
            status_frame,
            textvariable=self.elapsed_var,
            anchor="w",
        )
        elapsed_label.pack(fill="x", pady=(3, 0))

        log_frame = tk.Frame(status_frame)
        log_frame.pack(fill="x", pady=(7, 0))
        self.log_listbox = tk.Listbox(
            log_frame,
            height=4,
            font=("Menlo", 9),
            activestyle="none",
            relief="flat",
            bg="#fbfcfe",
            highlightthickness=1,
            highlightbackground="#d7dce3",
        )
        log_scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_listbox.yview,
        )
        self.log_listbox.configure(yscrollcommand=log_scrollbar.set)
        self.log_listbox.pack(side="left", fill="x", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        action_row = tk.Frame(outer, bg="#f5f6f8")
        action_row.pack(fill="x", pady=(12, 0))
        self.start_button = ttk.Button(action_row, text="开始处理", command=self._start)
        self.start_button.pack(side="right")
        self.cancel_button = ttk.Button(
            action_row,
            text="停止",
            command=self._cancel,
            state="disabled",
        )
        self.cancel_button.pack(side="right", padx=(0, 8))
        self.open_button = ttk.Button(
            action_row,
            text="打开结果文件夹",
            command=self._open_output,
            state="disabled",
        )
        self.open_button.pack(side="right", padx=(0, 8))

        status = tk.Label(
            outer,
            textvariable=self.status_var,
            bg="#f5f6f8",
            fg="#5b6472",
            font=("PingFang SC", 10),
            anchor="w",
            justify="left",
        )
        status.pack(fill="x", pady=(8, 0))

    def _set_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (
            self.add_files_button,
            self.add_folder_button,
            self.clear_button,
            self.output_button,
            self.start_button,
        ):
            widget.configure(state=state)
        self.cancel_button.configure(state=("disabled" if enabled else "normal"))
        combo_state = "readonly" if enabled else "disabled"
        self.model_combo.configure(state=combo_state)
        self.device_combo.configure(state=combo_state)
        self.recursive_check.configure(state=state)
        self.metadata_check.configure(state=state)
        self.output_entry.configure(state=state)

    def _set_progress_mode(self, mode: str) -> None:
        """Switch between an indeterminate busy indicator and a percentage."""

        if mode == "indeterminate":
            self.progress.configure(mode="indeterminate")
            try:
                self.progress.start(12)
            except Exception:
                # Some lightweight Tk themes do not implement animation; the
                # textual stage/count labels still provide useful feedback.
                pass
        else:
            try:
                self.progress.stop()
            except Exception:
                pass
            self.progress.configure(mode="determinate")

    def _set_counts(self, completed: int, failed: int, total: int) -> None:
        self._completed_count = max(0, int(completed))
        self._failed_count = max(0, int(failed))
        self._last_total = max(0, int(total))
        self.count_var.set(
            f"成功 {self._completed_count}  ·  失败 {self._failed_count}  ·  共 {self._last_total}"
        )

    def _set_elapsed(self, *, finished: bool = False) -> None:
        if self.started_at is None:
            self.elapsed_var.set("用时：—")
            return
        seconds = max(0.0, time.monotonic() - self.started_at)
        suffix = "（已完成）" if finished else ""
        self.elapsed_var.set(f"用时：{seconds:.1f} 秒{suffix}")

    def _append_log(self, message: str, *, kind: str = "info") -> None:
        """Append a short, auto-scrolling line to the visible activity log."""

        if not hasattr(self, "log_listbox"):
            return
        prefix = {"ok": "✓", "error": "✗", "info": "·"}.get(kind, "·")
        self.log_listbox.insert("end", f"{prefix} {message}")
        # Keep the log bounded so a very large folder does not grow the Tk
        # process without limit.  The aggregate count remains visible above.
        max_lines = 200
        while self.log_listbox.size() > max_lines:
            self.log_listbox.delete(0)
        self.log_listbox.see("end")

    def _reset_activity(self) -> None:
        self._set_counts(0, 0, 0)
        self._was_cancelled = False
        self.current_var.set("当前文件：—")
        self.stage_var.set("准备处理")
        self.progress_var.set(0.0)
        self.progress_percent_var.set("0%")
        self.elapsed_var.set("用时：0.0 秒")
        self._set_progress_mode("indeterminate")
        if hasattr(self, "log_listbox"):
            self.log_listbox.delete(0, "end")
        self._append_log("已提交任务，正在扫描图片…")

    def _destroy_window(self) -> None:
        """Destroy the Tk window once no worker callbacks can arrive."""

        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            # Tk may already have torn down its interpreter (for example when
            # Finder closes the launching terminal at the same time).
            pass

    def _close_after_worker(self) -> None:
        """Wait for the daemon worker to return before destroying Tk.

        ``done``/``error`` is queued just before the worker function returns,
        so a tiny race is possible if the window is destroyed immediately.
        Polling the thread state avoids abandoning a model object while it is
        still unwinding and keeps the close-confirmation promise precise.
        """

        if self._closed:
            return
        worker = self.worker
        if worker is not None and worker.is_alive():
            try:
                self.root.after(50, self._close_after_worker)
            except Exception:
                # If Tk is already unavailable, there is no UI left to keep
                # alive; mark it closed and let the worker finish in the
                # background rather than raising from the event loop.
                self._closed = True
            return
        self._destroy_window()

    def _add_paths(self, paths: Sequence[Path]) -> None:
        expanded = collect_image_paths(paths, recursive=self.recursive_var.get())
        existing = {str(path) for path in self.paths}
        for path in expanded:
            if str(path) not in existing:
                self.paths.append(path)
                existing.add(str(path))
                self.listbox.insert("end", str(path))
        if expanded:
            self.status_var.set(f"已选择 {len(self.paths)} 张图片。")
            self.stage_var.set(f"已准备 {len(self.paths)} 张图片")
            self.count_var.set(f"成功 0  ·  失败 0  ·  共 {len(self.paths)}")
        elif paths:
            self.status_var.set("没有找到支持的图片，请选择 JPG、PNG、WEBP、TIFF 或 BMP。")
            self.stage_var.set("未找到可处理图片")

    def _choose_files(self) -> None:
        names = self.filedialog.askopenfilenames(
            title="选择手机图片",
            filetypes=[
                ("图片", "*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp"),
                ("所有文件", "*"),
            ],
        )
        if names:
            self._add_paths([Path(name) for name in names])

    def _choose_folder(self) -> None:
        name = self.filedialog.askdirectory(title="选择图片文件夹")
        if name:
            self._add_paths([Path(name)])

    def _choose_output(self) -> None:
        name = self.filedialog.askdirectory(title="选择结果文件夹")
        if name:
            self.output_var.set(name)

    def _clear_paths(self) -> None:
        if self.running:
            return
        self.paths.clear()
        self.listbox.delete(0, "end")
        self.progress_var.set(0)
        self.progress_percent_var.set("0%")
        self._set_progress_mode("determinate")
        self._set_counts(0, 0, 0)
        self.current_var.set("当前文件：—")
        self.elapsed_var.set("用时：—")
        self.stage_var.set("等待选择图片")
        if hasattr(self, "log_listbox"):
            self.log_listbox.delete(0, "end")
        self.status_var.set("请选择图片或文件夹")

    def _start(self) -> None:
        if self.running:
            return
        if not self.paths:
            self.messagebox.showinfo("还没有图片", "请先添加图片或文件夹。")
            return
        output_dir = Path(self.output_var.get()).expanduser()
        if not str(output_dir).strip():
            self.messagebox.showerror("结果文件夹为空", "请选择一个结果文件夹。")
            return
        self.running = True
        self.cancel_event.clear()
        self._close_requested = False
        self.started_at = time.monotonic()
        self._reset_activity()
        self.open_button.configure(state="disabled")
        self._set_controls(False)
        self.output_status_var.set(f"输出位置：{output_dir.expanduser().resolve()}")
        self.status_var.set("正在准备；首次运行可能需要下载模型权重，请稍候…")
        args = (
            tuple(self.paths),
            output_dir,
            self.model_var.get().strip() or DEFAULT_MODEL,
            self.device_var.get().strip() or "auto",
            bool(self.recursive_var.get()),
            bool(self.metadata_var.get()),
        )
        self.worker = threading.Thread(target=self._run_worker, args=args, daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        if not self.running or self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.stage_var.set("正在停止…")
        self.status_var.set("正在停止；当前阶段/图片完成后会停止，请稍候…")
        self._append_log("已请求停止，等待当前阶段收尾…")

    def _on_close(self) -> None:
        """Keep a running worker from being abandoned accidentally."""

        if self.running:
            if self.cancel_event.is_set():
                # The user already confirmed a stop.  Remember the close
                # request and let the worker finish its current model call;
                # `_finish`/`_fail` will destroy the window without another
                # modal dialog.
                self._close_requested = True
                self.status_var.set("正在停止并关闭窗口，请稍候…")
                return
            try:
                confirmed = self.messagebox.askyesno(
                    "正在处理",
                    "当前还有图片正在处理。要停止并关闭窗口吗？",
                )
            except Exception:
                confirmed = False
            if confirmed:
                self._close_requested = True
                self._cancel()
            return
        self._destroy_window()

    def _status_callback(
        self,
        event: str,
        index: int,
        total: int,
        source: Optional[Path],
        result: Optional[BatchResult],
    ) -> None:
        """Forward worker status to the Tk thread; never touch widgets here."""

        self.events.put(("status", (event, index, total, source, result)))

    def _run_worker(
        self,
        paths: Sequence[Path],
        output_dir: Path,
        model: str,
        device: str,
        recursive: bool,
        save_metadata: bool,
    ) -> None:
        def progress(index: int, total: int, source: Path) -> None:
            self.events.put(("progress", (index, total, source)))

        try:
            results = process_files(
                paths,
                output_dir=output_dir,
                model_name=model,
                device=device,
                recursive=recursive,
                save_metadata=save_metadata,
                progress=progress,
                status=self._status_callback,
                should_stop=self.cancel_event.is_set,
            )
            self.events.put(("done", (results, output_dir)))
        except Exception as exc:
            # Queue only a lightweight type/message pair.  Retaining the raw
            # exception would also retain its traceback and all locals from a
            # failed model load, potentially keeping accelerator tensors alive
            # until the UI thread happened to consume the event.
            error_payload = (type(exc).__name__, str(exc))
            try:
                exc.__traceback__ = None
                exc.__cause__ = None
                exc.__context__ = None
            except Exception:
                pass
            del exc
            self.events.put(("error", error_payload))

    def _poll_events(self) -> None:
        if self._closed:
            return
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    index, total, source = payload
                    self.progress_var.set(float(index) / float(total) * 100.0)
                    self.progress_percent_var.set(f"{self.progress_var.get():.0f}%")
                    self.status_var.set(f"正在处理第 {index}/{total} 张：{source.name}")
                elif event == "status":
                    self._handle_status(*payload)
                elif event == "done":
                    self._finish(payload[0], payload[1])
                elif event == "error":
                    self._fail(payload)
        except queue.Empty:
            pass
        try:
            if not self._closed:
                self.root.after(120, self._poll_events)
        except Exception:
            self._closed = True

    def _handle_status(
        self,
        event: str,
        index: int,
        total: int,
        source: Optional[Path],
        result: Optional[BatchResult],
    ) -> None:
        """Render lifecycle events emitted by :func:`process_files`."""

        if total:
            self._last_total = int(total)
        if event == "scanning":
            self.stage_var.set("正在扫描图片…")
            self.status_var.set("正在扫描输入文件夹，请稍候…")
            self._set_progress_mode("indeterminate")
        elif event == "queued":
            self._set_counts(0, 0, total)
            self.stage_var.set("已找到图片，准备加载模型")
            self.status_var.set(f"已找到 {total} 张图片。")
            self._append_log(f"扫描完成，共 {total} 张图片。")
        elif event == "loading":
            self.stage_var.set("正在加载模型…")
            self.status_var.set("正在加载 BiRefNet；首次运行可能下载约 444 MB 权重…")
            self._set_progress_mode("indeterminate")
            self._append_log(
                f"正在加载模型（动态输入最长边上限 {DEFAULT_BIREFNET_MAX_SIDE} px）…"
            )
        elif event == "load_failed":
            self.stage_var.set("模型加载失败")
            self._set_progress_mode("determinate")
            self._append_log("模型加载失败。", kind="error")
        elif event == "ready":
            self.stage_var.set("模型已加载，开始处理")
            self.status_var.set("模型已加载，正在逐张处理…")
            self._set_progress_mode("determinate")
            self.progress_var.set(0.0)
            self.progress_percent_var.set("0%")
            self._append_log("模型已就绪，开始处理图片。", kind="ok")
        elif event == "processing" and source is not None:
            self.stage_var.set(f"正在处理第 {index}/{total} 张")
            self.current_var.set(f"当前文件：{source.name}")
            self.status_var.set(f"正在处理第 {index}/{total} 张：{source.name}")
            self.progress_var.set(float(max(0, index - 1)) / float(max(total, 1)) * 100.0)
            self.progress_percent_var.set(f"{self.progress_var.get():.0f}%")
            self._set_elapsed()
        elif event == "completed" and source is not None:
            self._set_counts(self._completed_count + 1, self._failed_count, total)
            self.progress_var.set(float(index) / float(max(total, 1)) * 100.0)
            self.progress_percent_var.set(f"{self.progress_var.get():.0f}%")
            self.current_var.set(f"当前文件：{source.name}")
            detail = ""
            if result is not None and result.info is not None:
                detail = f"（{result.info.output_size[0]}×{result.info.output_size[1]}）"
            self._append_log(f"完成 {source.name}{detail}", kind="ok")
            self._set_elapsed()
        elif event == "failed" and source is not None:
            self._set_counts(self._completed_count, self._failed_count + 1, total)
            self.progress_var.set(float(index) / float(max(total, 1)) * 100.0)
            self.progress_percent_var.set(f"{self.progress_var.get():.0f}%")
            error = result.error if result is not None else "未知错误"
            self.current_var.set(f"当前文件：{source.name}")
            self._append_log(f"失败 {source.name}：{error}", kind="error")
            self._set_elapsed()
        elif event == "cancelled":
            self._was_cancelled = True
            self.stage_var.set("已请求停止")
            self.status_var.set("正在整理已完成的结果…")
            self._append_log("批处理已停止。")
        elif event == "finished":
            self._set_progress_mode("determinate")
            self._set_elapsed(finished=True)
            self.stage_var.set("处理结束")

    def _finish(self, results: Sequence[BatchResult], output_dir: Path) -> None:
        self.running = False
        self._set_controls(True)
        succeeded = sum(item.succeeded for item in results)
        failed = len(results) - succeeded
        total = max(self._last_total, len(results))
        cancelled = self._was_cancelled
        close_requested = self._close_requested
        self._set_counts(succeeded, failed, total)
        self.progress_var.set(100 if not cancelled else (len(results) / float(total) * 100.0 if total else 0.0))
        self.progress_percent_var.set(f"{self.progress_var.get():.0f}%")
        self._set_progress_mode("determinate")
        self._set_elapsed(finished=True)
        self.open_button.configure(state="normal")
        self.output_status_var.set(f"输出位置：{output_dir.expanduser().resolve()}")
        if close_requested:
            # A confirmed close should not be followed by a modal completion
            # dialog.  Delay destruction by one Tk turn so the final state
            # update is applied and the worker thread has returned cleanly.
            self._close_after_worker()
            return
        if cancelled:
            self.stage_var.set("已停止")
            self.status_var.set(f"已停止：完成 {succeeded} 张，失败 {failed} 张。结果：{output_dir}")
            self._append_log(f"已停止，已保存 {succeeded} 张结果。")
            self.messagebox.showinfo(
                "处理已停止",
                f"已完成 {succeeded} 张，失败 {failed} 张。\n\n结果文件夹：{output_dir}",
            )
        elif failed:
            self.stage_var.set("完成，但有失败项目")
            failed_names = "\n".join(
                f"· {item.source.name}: {item.error}"
                for item in results
                if not item.succeeded
            )
            if len(failed_names) > 3000:
                failed_names = failed_names[:3000] + "\n…（其余失败项目请查看日志）"
            self.status_var.set(f"完成 {succeeded} 张，失败 {failed} 张。结果：{output_dir}")
            self.messagebox.showwarning(
                "处理完成，但有失败项目",
                f"成功 {succeeded} 张，失败 {failed} 张。\n\n{failed_names}",
            )
        else:
            self.stage_var.set("全部完成")
            self.status_var.set(f"已完成 {succeeded} 张。结果：{output_dir}")
            self.messagebox.showinfo("处理完成", f"已输出 {succeeded} 张透明 PNG。\n\n结果文件夹：{output_dir}")

    def _fail(self, exc: Any) -> None:
        self.running = False
        self._set_controls(True)
        self._set_progress_mode("determinate")
        self._set_elapsed(finished=True)
        self.stage_var.set("启动失败")
        self.status_var.set("启动失败，请检查依赖和网络连接。")
        if isinstance(exc, tuple) and len(exc) == 2:
            error_name, error_message = str(exc[0]), str(exc[1])
        else:
            # Keep compatibility with callers/tests that may still inject an
            # exception object directly into the event queue.
            error_name, error_message = type(exc).__name__, str(exc)
        self._append_log(f"启动失败：{error_name}: {error_message}", kind="error")
        if self._close_requested:
            self._close_after_worker()
            return
        self.messagebox.showerror(
            "无法开始处理",
            f"{error_name}: {error_message}\n\n"
            "首次运行请保持网络可用，以便下载 BiRefNet 权重；"
            "也可以双击“手机图片一键处理.command”自动安装依赖。",
        )

    def _open_output(self) -> None:
        path = Path(self.output_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            # Fallback keeps the button useful on unusual desktop setups.
            webbrowser.open(path.as_uri())


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        import tkinter as tk
    except ImportError as exc:
        print(
            "当前 Python 没有 Tk 图形界面组件。请使用 macOS 的“手机图片一键处理.command”，"
            "或直接运行 scripts/batch_normalize.py。",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description="手机商品图一键处理窗口")
    parser.add_argument("paths", nargs="*", type=Path, help="可选的初始图片路径")
    args = parser.parse_args(argv)
    root = tk.Tk()
    NormalizerWindow(root, args.paths, auto_start=bool(args.paths))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
