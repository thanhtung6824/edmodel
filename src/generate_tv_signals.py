"""Convert signals JSON (from validate scripts) to TradingView Pine Script.

Usage:
    python -m src.validate_4h                        # generates signals_4h.json
    python -m src.generate_tv_signals signals_4h.json  # generates signals_4h.pine

    python -m src.validate_1h                        # generates signals_1h.json
    python -m src.generate_tv_signals signals_1h.json  # generates signals_1h.pine
"""

import json
import sys


def generate_pine(data):
    """Generate Pine Script v5 indicator from signal data."""
    timeframe = data["timeframe"]
    threshold = data["threshold"]
    signals = data["signals"]

    n_wins = sum(s["result"] == 1 for s in signals)
    n_losses = len(signals) - n_wins
    n_long = sum(s["dir"] == 1 for s in signals)
    n_short = len(signals) - n_long
    win_rate = n_wins / len(signals) * 100 if signals else 0

    L = []
    L.append('//@version=5')
    L.append(f'indicator("SFP Signals ({timeframe})", overlay=true, max_labels_count=500, max_lines_count=500)')
    L.append('')
    L.append(f'// Signals: {len(signals)} ({n_long} long, {n_short} short) | Win rate: {win_rate:.0f}% | Threshold: {threshold}')
    L.append(f'// IMPORTANT: Use on {timeframe} BTCUSDT chart')
    L.append('')
    L.append('show_tp_sl = input.bool(true, "Show TP/SL levels")')
    L.append('show_stats = input.bool(true, "Show Stats Table")')
    L.append('')

    # Signal arrays
    L.append('// --- Signal Data ---')
    L.append('var sig_time = array.new_int()')
    L.append('var sig_dir = array.new_int()')
    L.append('var sig_entry = array.new_float()')
    L.append('var sig_tp = array.new_float()')
    L.append('var sig_sl = array.new_float()')
    L.append('var sig_ratio = array.new_float()')
    L.append('var sig_result = array.new_int()')
    # Split into chunks of 150 signals per block (150*7=1050 vars, under Pine's 1200 limit)
    CHUNK = 150
    for chunk_start in range(0, len(signals), CHUNK):
        chunk = signals[chunk_start : chunk_start + CHUNK]
        L.append('')
        L.append('if barstate.isfirst')
        for s in chunk:
            L.append(f'    array.push(sig_time, {s["time_ms"]})')
            L.append(f'    array.push(sig_dir, {s["dir"]})')
            L.append(f'    array.push(sig_entry, {s["entry"]:.2f})')
            L.append(f'    array.push(sig_tp, {s["tp_price"]:.2f})')
            L.append(f'    array.push(sig_sl, {s["sl_price"]:.2f})')
            L.append(f'    array.push(sig_ratio, {s["ratio"]:.4f})')
            L.append(f'    array.push(sig_result, {s["result"]})')

    # Plot logic — pointer advances through sorted signals
    L.append('')
    L.append('// --- Plot Logic ---')
    L.append('var int ptr = 0')
    L.append('var int total_wins = 0')
    L.append('var int total_losses = 0')
    L.append('')
    # Timeframe in ms for TP/SL line extension
    tf_ms = {'4h': 14400000, '1h': 3600000, '15min': 900000}
    bar_ms = tf_ms.get(timeframe, 3600000)
    line_end_bars = 12  # extend TP/SL lines 12 bars ahead

    L.append('if ptr < array.size(sig_time)')
    L.append('    if time >= array.get(sig_time, ptr)')
    L.append('        int sig_t = array.get(sig_time, ptr)')
    L.append('        int d = array.get(sig_dir, ptr)')
    L.append('        float entry = array.get(sig_entry, ptr)')
    L.append('        float tp_val = array.get(sig_tp, ptr)')
    L.append('        float sl_val = array.get(sig_sl, ptr)')
    L.append('        float r = array.get(sig_ratio, ptr)')
    L.append('        int w = array.get(sig_result, ptr)')
    L.append('        color clr = w == 1 ? color.green : color.red')
    L.append('        string txt = str.tostring(r, "#.##") + (w == 1 ? " W" : " L")')
    L.append('        if d == 1')
    L.append('            label.new(sig_t, low, "L " + txt, xloc=xloc.bar_time, style=label.style_label_up, color=clr, textcolor=color.white, size=size.small)')
    L.append('        else')
    L.append('            label.new(sig_t, high, "S " + txt, xloc=xloc.bar_time, style=label.style_label_down, color=clr, textcolor=color.white, size=size.small)')
    L.append('        if show_tp_sl')
    L.append(f'            line.new(sig_t, tp_val, sig_t + {bar_ms * line_end_bars}, tp_val, xloc=xloc.bar_time, color=color.new(color.green, 40), style=line.style_dashed, width=1)')
    L.append(f'            line.new(sig_t, sl_val, sig_t + {bar_ms * line_end_bars}, sl_val, xloc=xloc.bar_time, color=color.new(color.red, 40), style=line.style_dashed, width=1)')
    L.append(f'            line.new(sig_t, entry, sig_t + {bar_ms * line_end_bars}, entry, xloc=xloc.bar_time, color=color.new(color.blue, 40), style=line.style_dotted, width=1)')
    L.append('        if w == 1')
    L.append('            total_wins += 1')
    L.append('        else')
    L.append('            total_losses += 1')
    L.append('        ptr += 1')

    # Stats table
    L.append('')
    L.append('// --- Stats Table ---')
    L.append('if show_stats and barstate.islast')
    L.append('    int total = total_wins + total_losses')
    L.append('    float wr = total > 0 ? total_wins / total * 100.0 : 0.0')
    L.append('    var t = table.new(position.top_right, 2, 4, bgcolor=color.new(color.black, 80), border_color=color.gray, border_width=1)')
    L.append('    table.cell(t, 0, 0, "Trades", text_color=color.white, text_size=size.small)')
    L.append('    table.cell(t, 1, 0, str.tostring(total), text_color=color.white, text_size=size.small)')
    L.append('    table.cell(t, 0, 1, "Wins", text_color=color.green, text_size=size.small)')
    L.append('    table.cell(t, 1, 1, str.tostring(total_wins), text_color=color.green, text_size=size.small)')
    L.append('    table.cell(t, 0, 2, "Losses", text_color=color.red, text_size=size.small)')
    L.append('    table.cell(t, 1, 2, str.tostring(total_losses), text_color=color.red, text_size=size.small)')
    L.append('    table.cell(t, 0, 3, "Win Rate", text_color=color.white, text_size=size.small)')
    L.append('    table.cell(t, 1, 3, str.tostring(wr, "#.0") + "%", text_color=wr >= 50 ? color.green : color.red, text_size=size.small)')

    return '\n'.join(L)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.generate_tv_signals <signals_file.json>")
        print("Example: python -m src.generate_tv_signals signals_4h.json")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file) as f:
        data = json.load(f)

    signals = data["signals"]
    timeframe = data["timeframe"]

    if not signals:
        print("No signals in file.")
        return

    n_wins = sum(s["result"] == 1 for s in signals)
    print(f"Loaded {len(signals)} signals from {input_file}")
    print(f"  Timeframe: {timeframe} | Wins: {n_wins} | Losses: {len(signals) - n_wins}")

    pine_code = generate_pine(data)

    output_file = input_file.replace(".json", ".pine")
    with open(output_file, "w") as f:
        f.write(pine_code)

    print(f"\nSaved: {output_file}")
    print(f"\nHow to use:")
    print(f"  1. Open TradingView -> BTCUSDT {timeframe} chart")
    print(f"  2. Pine Editor -> New Script -> paste contents of {output_file}")
    print(f"  3. Click 'Add to Chart'")


main()
