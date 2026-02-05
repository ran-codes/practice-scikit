export interface PythonResult {
  success: boolean;
  html?: string;
  error?: string;
}

export async function runPythonCode(
  pyodide: any,
  code: string
): Promise<PythonResult> {
  try {
    // Setup stdout capture and matplotlib handling
    await pyodide.runPythonAsync(`
import sys
from io import StringIO, BytesIO
import base64
_stdout_capture = StringIO()
_old_stdout = sys.stdout
sys.stdout = _stdout_capture
_matplotlib_images = []

# Capture matplotlib figures
def _capture_matplotlib():
    import matplotlib.pyplot as plt
    global _matplotlib_images
    _matplotlib_images = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        _matplotlib_images.append(img_base64)
        plt.close(fig)
`);

    // Run user code
    await pyodide.runPythonAsync(code);

    // Collect output
    const html = await pyodide.runPythonAsync(`
sys.stdout = _old_stdout
_printed = _stdout_capture.getvalue()
_result_value = globals().get('result')

# Capture any matplotlib figures
try:
    _capture_matplotlib()
except:
    pass

_html_parts = []

# Add matplotlib images first
for img_b64 in _matplotlib_images:
    _html_parts.append(f'<div class="matplotlib-output"><img src="data:image/png;base64,{img_b64}" alt="matplotlib figure" /></div>')

if _printed.strip():
    import html as html_mod
    _html_parts.append('<pre class="output-print">' + html_mod.escape(_printed) + '</pre>')

if _result_value is not None:
    if hasattr(_result_value, '_repr_html_'):
        _html_parts.append(_result_value._repr_html_())
    else:
        import html as html_mod
        _html_parts.append('<pre>' + html_mod.escape(str(_result_value)) + '</pre>')

''.join(_html_parts) if _html_parts else '<p class="text-muted-foreground italic">No output</p>'
`);

    return { success: true, html };
  } catch (error) {
    // Reset stdout on error
    try {
      await pyodide.runPythonAsync(
        "sys.stdout = _old_stdout if '_old_stdout' in dir() else sys.stdout"
      );
    } catch {
      // Ignore cleanup errors
    }

    const errorMessage =
      error instanceof Error ? error.message : String(error);
    return { success: false, error: errorMessage };
  }
}
