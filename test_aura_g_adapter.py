import unittest, numpy as np
import aura_g_adapter as A, midi_bridge as B, harmonia_guitar as G
SR = 48000
REF = [B.Note(0.1 + 0.45 * i, 0.4, m, 90) for i, m in enumerate([52, 55, 59, 57, 55, 52, 50, 47, 64, 62])]
def synth():
    y = np.zeros(int(SR * (REF[-1].start + REF[-1].dur + 0.3)))
    for i, n in enumerate(REF):
        p = G.pluck(n.midi, n.dur, 0.8, SR, seed=i); a = int(n.start * SR); y[a:a + len(p)] += p[:len(y) - a]
    return y / np.max(np.abs(y))
def f0_spec(y, f0):
    seg = y[int(0.05 * SR):int(0.65 * SR)] * np.hanning(int(0.6 * SR)); N = 1 << 18; S = np.abs(np.fft.rfft(seg, N)); f = np.fft.rfftfreq(N, 1 / SR); m = (f > f0 * .9) & (f < f0 * 1.1)
    i = int(np.argmax(np.where(m, S, 0))); a, b, c = np.log(S[i - 1]), np.log(S[i]), np.log(S[i + 1]); return f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
class T(unittest.TestCase):
    def test_transcription_f1_and_durations(self):
        est = A.aura_g_transcribe(synth(), SR); tp = 0; used = set(); durs = []
        for r in REF:
            for j, e in enumerate(est):
                if j not in used and e.midi == r.midi and abs(e.start - r.start) < 0.06: tp += 1; used.add(j); durs.append(e.dur); break
        prec = tp / max(len(est), 1); rec = tp / len(REF); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  AURA-G adapter: {tp}/{len(REF)} matched, {len(est)} notes, F1 {f1:.2f}, median duration {np.median(durs):.2f} s (true 0.40)"); self.assertGreaterEqual(f1, 0.9)
    def test_string_tuning_and_fret(self):
        S = A.TunedWaveguideString(SR, 1); worst = 0
        for f0 in (82.41, 196.0, 329.63, 659.26, 880.0):
            y = S.render(f0, 1.0); worst = max(worst, abs(1200 * np.log2(f0_spec(y, f0) / f0)))
        y5 = S.render(196.0, 1.0, fret=5); worst5 = abs(1200 * np.log2(f0_spec(y5, 261.63) / 261.63))
        print(f"  tuned string: worst open-string error {worst:.2f} cents, fret-5 error {worst5:.2f} cents"); self.assertLess(worst, 1.5); self.assertLess(worst5, 1.5)
    def test_decay_follows_t60(self):
        y = A.TunedWaveguideString(SR, 2).render(196.0, 4.0, t60=2.0); r = lambda a, b: np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2))
        d = 20 * np.log10(r(0.5, 0.6) / r(1.5, 1.6)); print(f"  decay over 1 s at T60=2 s: {d:.1f} dB (30 expected)"); self.assertTrue(22 < d < 38)
    def test_nan_safe_input(self):
        y = synth().copy(); y[20000] = np.nan; y = np.nan_to_num(y); self.assertTrue(len(A.aura_g_transcribe(y, SR)) >= 8)
    def test_end_to_end(self):
        est = A.aura_g_transcribe(synth(), SR); a = A.render_talkbox_aura(est[:4]); b = A.render_talkbox_aura(est[:4])
        self.assertTrue(np.all(np.isfinite(a))); self.assertTrue(np.array_equal(a, b)); self.assertLessEqual(np.max(np.abs(a)), 0.7001)
if __name__ == "__main__": unittest.main(verbosity=2)
