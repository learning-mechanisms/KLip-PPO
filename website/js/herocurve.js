"use strict";
/* hero: CartPole-v1 split player (PPO-Clip and per-sample KL), play/pause + scrub */
(function () {
  var svg = document.getElementById("heroCurve");
  var C = window.CURVES;
  if (!svg || !C || !C["CartPole-v1"]) return;
  var cp = C["CartPole-v1"];
  var clip = (cp.ppo_clip || []).filter(function (p) {
    return isFinite(p[1]);
  });
  var ps = (cp.ppo_kl_per_sample || []).filter(function (p) {
    return isFinite(p[1]);
  });
  if (clip.length < 2 || ps.length < 2) return;

  var playBtn = document.getElementById("heroPlay");
  var scrub = document.getElementById("heroScrub");
  var stepEl = document.getElementById("heroStep");

  var NS = "http://www.w3.org/2000/svg";
  function mk(t, a) {
    var n = document.createElementNS(NS, t);
    for (var k in a) n.setAttribute(k, a[k]);
    return n;
  }
  function txt(a, s) {
    var t = mk("text", a);
    t.setAttribute("font-family", "Inter, system-ui, sans-serif");
    t.textContent = s;
    return t;
  }

  var COL = { clip: "#2f43c4", ps: "#16181d", grid: "#e9ecf0", dim: "#8a8f98", head: "#2f43c4" };
  var W = 380,
    H = 196,
    mt = 24,
    mb = 30,
    ml = 36,
    mr = 10,
    gut = 22;
  var pw = (W - ml - mr - gut) / 2;
  var PAN = [
    { x0: ml, data: clip, title: "PPO-Clip", col: COL.clip, dash: false, showY: true },
    { x0: ml + pw + gut, data: ps, title: "per-sample KL", col: COL.ps, dash: true, showY: false },
  ];

  var ylo = Infinity,
    yhi = -Infinity,
    xmax = 0;
  [clip, ps].forEach(function (d) {
    d.forEach(function (pt) {
      ylo = Math.min(ylo, pt[1]);
      yhi = Math.max(yhi, pt[1]);
      xmax = Math.max(xmax, pt[0]);
    });
  });
  var pad = (yhi - ylo) * 0.08 || 1;
  ylo -= pad;
  yhi += pad;
  if (!xmax) xmax = 1;
  var ys = function (v) {
    return mt + (1 - (v - ylo) / (yhi - ylo)) * (H - mt - mb);
  };

  function render(p) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var ymid = (mt + (H - mb)) / 2;
    svg.appendChild(txt({ x: 11, y: ymid, "text-anchor": "middle", "font-size": 10, fill: COL.dim, transform: "rotate(-90 11 " + ymid + ")" }, "return"));

    PAN.forEach(function (P) {
      var xs = function (s) {
        return P.x0 + (s / xmax) * pw;
      };
      for (var i = 0; i <= 4; i++) {
        var v = ylo + ((yhi - ylo) * i) / 4,
          y = ys(v);
        svg.appendChild(mk("line", { x1: P.x0, y1: y, x2: P.x0 + pw, y2: y, stroke: COL.grid, "stroke-width": 1 }));
        if (P.showY) svg.appendChild(txt({ x: P.x0 - 5, y: y + 3, "text-anchor": "end", "font-size": 9, fill: COL.dim }, Math.round(v)));
      }
      svg.appendChild(mk("line", { x1: P.x0, y1: H - mb, x2: P.x0 + pw, y2: H - mb, stroke: "#9aa0a8", "stroke-width": 1.3 }));
      svg.appendChild(mk("line", { x1: P.x0, y1: mt, x2: P.x0, y2: H - mb, stroke: "#9aa0a8", "stroke-width": 1.3 }));
      svg.appendChild(txt({ x: P.x0 + pw / 2, y: mt - 9, "text-anchor": "middle", "font-size": 11, fill: P.col }, P.title));
      svg.appendChild(txt({ x: P.x0 + pw, y: H - mb + 15, "text-anchor": "end", "font-size": 9, fill: COL.dim }, "steps →"));

      var pts = P.data,
        fpos = p * (pts.length - 1),
        ki = Math.floor(fpos),
        frac = fpos - ki,
        d = "",
        i2;
      for (i2 = 0; i2 <= ki; i2++) d += (d ? "L" : "M") + xs(pts[i2][0]).toFixed(1) + " " + ys(pts[i2][1]).toFixed(1);
      var hx, hy;
      if (ki < pts.length - 1) {
        hx = pts[ki][0] + (pts[ki + 1][0] - pts[ki][0]) * frac;
        hy = pts[ki][1] + (pts[ki + 1][1] - pts[ki][1]) * frac;
        d += "L" + xs(hx).toFixed(1) + " " + ys(hy).toFixed(1);
      } else {
        hx = pts[ki][0];
        hy = pts[ki][1];
      }
      svg.appendChild(
        mk("path", { d: d, fill: "none", stroke: P.col, "stroke-width": P.dash ? 1.8 : 2.3, "stroke-dasharray": P.dash ? "5 4" : "0", "stroke-linejoin": "round", "stroke-linecap": "round" }),
      );
      svg.appendChild(mk("circle", { cx: xs(hx), cy: ys(hy), r: 3, fill: P.col, stroke: "#fff", "stroke-width": 1 }));
      var px = P.x0 + p * pw;
      svg.appendChild(mk("line", { x1: px, y1: mt, x2: px, y2: H - mb, stroke: COL.head, "stroke-width": 1, "stroke-dasharray": "2 3", opacity: 0.55 }));
    });
    if (scrub) scrub.value = p;
    if (stepEl) stepEl.textContent = Math.round((p * xmax) / 1000) + "k / " + Math.round(xmax / 1000) + "k steps";
  }

  var p = 0,
    playing = false,
    raf = null,
    last = 0;
  function loop(ts) {
    if (!playing) return;
    if (!last) last = ts;
    p += (ts - last) / 8000;
    last = ts; // ~8s for a full pass
    if (p >= 1) {
      p = 0;
      last = ts;
    } // loop back to the start
    render(p);
    raf = requestAnimationFrame(loop);
  }
  if (playBtn)
    playBtn.onclick = function () {
      if (playing) {
        playing = false;
        playBtn.textContent = "play";
        cancelAnimationFrame(raf);
      } else {
        playing = true;
        playBtn.textContent = "pause";
        last = 0;
        raf = requestAnimationFrame(loop);
      }
    };
  if (scrub)
    scrub.addEventListener("input", function () {
      playing = false;
      if (playBtn) playBtn.textContent = "play";
      cancelAnimationFrame(raf);
      p = +scrub.value;
      render(p);
    });

  render(0);
  if (!(window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches)) {
    var io = new IntersectionObserver(
      function (es) {
        es.forEach(function (e) {
          if (e.isIntersecting && p === 0 && !playing && playBtn) {
            playBtn.click();
            io.disconnect();
          }
        });
      },
      { threshold: 0.3 },
    );
    io.observe(svg);
  }
})();
