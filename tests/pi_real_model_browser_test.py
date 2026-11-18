"""Pi with real models in Chrome: each model gets the same small file task through the terminal.

Opens each GGUF through the page, runs `pi Write the text "<word>" to a file named <file>`, and
checks with `cat` whether the file holds the word. Small models often fail tool tasks, so this
reports per model whether a tool call in the model's own syntax ran and the file is right; it
fails only when the page errors or when no model completes the task.

    python3 tests/pi_real_model_browser_test.py --runtime dist model.gguf [model.gguf ...]
"""
import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from browser_test import FLAGS, serve


def run(page, text, timeout):
    page.locator('#terminal-input').fill(text)
    page.locator('.terminal-form button[type=submit]').click()
    page.wait_for_function("() => !document.querySelector('.terminal-form button[type=submit]').disabled", timeout=timeout)
    return page.locator('.terminal-log').inner_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('models', type=Path, nargs='+')
    a = ap.parse_args()
    server = serve(a.runtime)
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', args=FLAGS)
        for i, model in enumerate(a.models):
            context = browser.new_context(viewport={'width': 1280, 'height': 900})
            context.add_init_script("if(window===window.top)localStorage.setItem('evolve-llm-settings',JSON.stringify({temperature:0,maxTokens:512,ctx:4096,thinking:false,gpu:'high-performance',gpuMemory:'auto'}))")
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/')
            page.wait_for_selector('#gpu-chip[data-state=ok]', timeout=30000)
            page.set_input_files('#file-input', str(model))
            page.wait_for_selector('#ready:not([hidden])', timeout=300000)
            page.locator('#open-terminal').click()
            page.wait_for_function("() => document.querySelector('#terminal-status')?.textContent.startsWith('Ready')", timeout=60000)
            word, name = f'evolve{i}ok', f'note{i}.txt'
            log = run(page, f'pi Write the text "{word}" to a file named {name}', 600000)
            ran = 'Pi → write' in log or 'Pi → bash' in log
            content = run(page, f'cat {name}', 60000).rsplit(f'cat {name}', 1)[-1]
            ok = word in content
            results.append((model.name, ran, ok, errors))
            print(f'{model.name}: tool call ran: {ran}; file holds the text: {ok}' + (f'; page errors: {errors}' if errors else ''), flush=True)
            context.close()
        browser.close()
    if any(e for *_, e in results):
        sys.exit('page errors')
    if not any(ok for _, _, ok, _ in results):
        sys.exit('no model completed the task')


if __name__ == '__main__':
    main()
