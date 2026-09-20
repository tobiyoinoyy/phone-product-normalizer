-- macOS 原生轻量 GUI：由“手机图片一键处理.command”调用。
-- 选择图片或文件夹后，交给同目录的 batch_normalize.py 处理。

on run argv
	if (count of argv) < 3 then
		display dialog "启动参数不完整，请从项目目录双击“手机图片一键处理.command”。" buttons {"好"} default button "好"
		return
	end if

	set pythonPath to item 1 of argv
	set batchPath to item 2 of argv
	set outputPath to item 3 of argv

	set selectionMode to ""
	try
		set modeChoice to display dialog ¬
			"手机商品图一键处理\n\n默认模型：BiRefNet_dynamic\n输出：透明 PNG + JSON 记录" ¬
			buttons {"取消", "选择文件夹", "选择图片"} ¬
			default button "选择图片" cancel button "取消"
		set selectionMode to button returned of modeChoice
	 on error number -128
		return
	end try

	set selectedItems to {}
	try
		if selectionMode is "选择文件夹" then
			set selectedFolder to choose folder with prompt "选择要处理的图片文件夹（会包含子文件夹）"
			set selectedItems to {selectedFolder}
		else if selectionMode is "选择图片" then
			set selectedItems to choose file with prompt ¬
				"选择手机图片（可按住 Command 多选）" ¬
				with multiple selections allowed
		else
			return
		end if
	 on error number -128
		return
	end try

	if (count of selectedItems) is 0 then return

	set commandLine to (quoted form of pythonPath) & " " & (quoted form of batchPath) & ¬
		" --output-dir " & (quoted form of outputPath)
	repeat with selectedItem in selectedItems
		set commandLine to commandLine & " " & (quoted form of (POSIX path of (contents of selectedItem)))
	end repeat

	try
		display notification "正在处理图片，首次运行可能需要下载模型权重…" with title "手机商品图一键处理"
	on error
		-- Older macOS versions may not support notifications; processing can continue.
	end try

	try
		set resultText to do shell script commandLine
		try
			display notification "处理完成，透明 PNG 已保存到结果文件夹。" with title "手机商品图一键处理"
		end try
		display dialog "处理完成。\n\n结果文件夹：" & outputPath ¬
			buttons {"打开结果文件夹", "稍后查看"} default button "打开结果文件夹"
		if button returned of result is "打开结果文件夹" then
			tell application "Finder" to open POSIX file outputPath
		end if
	on error errText number errNum
		try
			display notification "处理失败，请查看终端中的详细信息。" with title "手机商品图一键处理"
		end try
		display dialog "处理未能完成。\n\n" & errText ¬
			buttons {"好"} default button "好"
	end try
end run
