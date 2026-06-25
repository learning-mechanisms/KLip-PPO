/* player.js - learning-curve player (experiments).
   data-mode="split" draws PPO-Clip and per-sample KL side by side; default overlays all four. */
(function curvePlayer() {
  "use strict";
  const plot = document.getElementById("curvePlot");
  if (!plot || !window.CURVES) return;
  const C = window.CURVES;
  const tasksBox = document.getElementById("tasks");
  const legendBox = document.getElementById("curveLegend");
  const playBtn = document.getElementById("playBtn");
  const scrub = document.getElementById("scrub");
  const stepRead = document.getElementById("stepReadout");
  const SPLIT = plot.dataset.mode === "split";

  const NS = "http://www.w3.org/2000/svg";
  const mk = (t, a) => {
    const n = document.createElementNS(NS, t);
    for (const k in a) n.setAttribute(k, a[k]);
    return n;
  };
  const txt = (a, s) => {
    const t = mk("text", Object.assign({ "font-family": "Inter, system-ui, sans-serif" }, a));
    t.textContent = s;
    return t;
  };
  const COL = { clip: "#2f43c4", ps: "#16181d", fix: "#d08a2c", adp: "#1f8a5b", kill: "#c0392b", dim: "#5b626e", grid: "#e9ecf0" };
  const W = 760,
    H = 360,
    mt = 22,
    mb = 34;

  const ENVS = ["Ant-v4", "Humanoid-v4", "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"];
  // every drawable series: key, label, colour, dashed?
  const S = {
    clip: ["ppo_clip", "PPO-Clip", COL.clip, false],
    ps: ["ppo_kl_per_sample", "per-sample KL", COL.ps, true],
    fix: ["ppo_kl_fixed", "fixed β", COL.fix, false],
    adp: ["ppo_kl_adaptive", "adaptive β", COL.adp, false],
  };
  // single mode overlays all four; split mode is clip vs per-sample
  const ALL = [S.clip, S.ps, S.fix, S.adp];
  const PANELS = SPLIT ? [[S.clip], [S.ps]] : [ALL];
  const SCOPE = SPLIT ? [S.clip, S.ps] : ALL; // which series set the shared y-range

  const REDUCE = matchMedia("(prefers-reduced-motion:reduce)").matches;
  let env = ENVS[0],
    p = 0,
    playing = false,
    raf = null,
    last = 0,
    holdUntil = 0, // while set, we rest on a finished curve until this timestamp, then roll to the next task
    fading = false; // true during the cross-fade hand-off between carousel tasks
  const clean = (k) => ((C[env] && C[env][k]) || []).filter((pt) => Number.isFinite(pt[1]));

  // highlight the active task and rewind playback to its start
  function selectEnv(e) {
    env = e;
    const i = ENVS.indexOf(e);
    [...tasksBox.children].forEach((x, j) => x.classList.toggle("on", j === i));
    p = 0;
    scrub.value = 0;
    render();
  }

  // smooth carousel hand-off: fade the plot out, swap the task while hidden, fade back in
  const FADE = 220;
  function crossfadeTo(e) {
    if (REDUCE) {
      selectEnv(e);
      last = 0;
      return;
    }
    fading = true;
    plot.style.transition = "opacity " + FADE + "ms ease";
    plot.style.opacity = "0";
    setTimeout(() => {
      selectEnv(e); // redraw at the start while hidden
      requestAnimationFrame(() => (plot.style.opacity = "1"));
      setTimeout(() => {
        fading = false;
        last = 0; // re-base loop timing so the new task starts fresh
      }, FADE);
    }, FADE);
  }

  // task selector (shared across panels); clicking a task plays it and the carousel resumes from there
  ENVS.forEach((e) => {
    const b = document.createElement("button");
    b.textContent = e.replace(/-v\d$/, "");
    if (e === env) b.classList.add("on");
    b.onclick = () => {
      selectEnv(e);
      holdUntil = 0;
      last = 0;
      if (!playing) {
        playing = true;
        playBtn.textContent = "pause";
        raf = requestAnimationFrame(loop);
      }
    };
    tasksBox.appendChild(b);
  });

  function bounds() {
    let ylo = Infinity,
      yhi = -Infinity,
      xmax = 0;
    SCOPE.forEach(([k]) =>
      clean(k).forEach(([s, v]) => {
        ylo = Math.min(ylo, v);
        yhi = Math.max(yhi, v);
        xmax = Math.max(xmax, s);
      }),
    );
    if (!isFinite(ylo)) {
      ylo = 0;
      yhi = 1;
    }
    const pad = (yhi - ylo) * 0.08 || 1;
    return { ylo: ylo - pad, yhi: yhi + pad, xmax: xmax || 1 };
  }

  // draw one panel into [x0, x0+pw]; ys maps return -> y (shared); returns live values
  function drawPanel(x0, pw, series, title, showY, B, ys) {
    const { xmax } = B;
    const xs = (s) => x0 + (s / xmax) * pw;
    // frame + gridlines
    plot.appendChild(mk("line", { x1: x0, y1: H - mb, x2: x0 + pw, y2: H - mb, stroke: COL.ps, "stroke-width": 1.5 }));
    plot.appendChild(mk("line", { x1: x0, y1: mt, x2: x0, y2: H - mb, stroke: COL.ps, "stroke-width": 1.5 }));
    for (let i = 0; i <= 4; i++) {
      const v = B.ylo + ((B.yhi - B.ylo) * i) / 4,
        y = ys(v);
      plot.appendChild(mk("line", { x1: x0, y1: y, x2: x0 + pw, y2: y, stroke: COL.grid, "stroke-width": 1 }));
      if (showY) plot.appendChild(txt({ x: x0 - 6, y: y + 3, "text-anchor": "end", "font-size": 10, fill: COL.dim }, Math.round(v)));
    }
    if (title) plot.appendChild(txt({ x: x0 + pw / 2, y: mt - 8, "text-anchor": "middle", "font-size": 12, fill: COL.ps }, title));
    plot.appendChild(txt({ x: x0 + pw, y: H - mb + 16, "text-anchor": "end", "font-size": 10, fill: COL.dim }, "env steps →"));

    const live = [];
    series.forEach(([k, label, col, dash]) => {
      const pts = clean(k);
      if (pts.length < 2) return;
      const fpos = p * (pts.length - 1),
        ki = Math.floor(fpos),
        frac = fpos - ki;
      let d = "";
      for (let i = 0; i <= ki; i++) d += (d ? "L" : "M") + xs(pts[i][0]).toFixed(1) + " " + ys(pts[i][1]).toFixed(1);
      let hx, hy;
      if (ki < pts.length - 1) {
        hx = pts[ki][0] + (pts[ki + 1][0] - pts[ki][0]) * frac;
        hy = pts[ki][1] + (pts[ki + 1][1] - pts[ki][1]) * frac;
        d += "L" + xs(hx).toFixed(1) + " " + ys(hy).toFixed(1);
      } else {
        hx = pts[ki][0];
        hy = pts[ki][1];
      }
      plot.appendChild(mk("path", { d, fill: "none", stroke: col, "stroke-width": dash ? 2 : 2.6, "stroke-dasharray": dash ? "6 5" : "0", "stroke-linejoin": "round", "stroke-linecap": "round" }));
      plot.appendChild(mk("circle", { cx: xs(hx), cy: ys(hy), r: 3.2, fill: col, stroke: "#fff", "stroke-width": 1 }));
      live.push([label, col, dash, hy]);
    });
    // playhead
    const px = x0 + p * pw;
    plot.appendChild(mk("line", { x1: px, y1: mt, x2: px, y2: H - mb, stroke: COL.kill, "stroke-width": 1, "stroke-dasharray": "2 3", opacity: 0.6 }));
    return live;
  }

  function render() {
    const B = bounds();
    const ys = (v) => mt + (1 - (v - B.ylo) / (B.yhi - B.ylo)) * (H - mt - mb);
    plot.innerHTML = "";
    // shared y-axis title on the far left
    const ymid = (mt + (H - mb)) / 2;
    plot.appendChild(txt({ x: 13, y: ymid, "text-anchor": "middle", "font-size": 11, fill: COL.dim, transform: "rotate(-90 13 " + ymid + ")" }, "episode return"));

    const ml = 48,
      mr = 14,
      gut = 30;
    let live = [];
    if (PANELS.length === 1) {
      live = drawPanel(ml, W - ml - mr, PANELS[0], null, true, B, ys);
    } else {
      const pw = (W - ml - mr - gut) / 2;
      live = drawPanel(ml, pw, PANELS[0], PANELS[0][0][1], true, B, ys).concat(drawPanel(ml + pw + gut, pw, PANELS[1], PANELS[1][0][1], false, B, ys));
    }

    // live legend + step readout
    if (legendBox) {
      legendBox.innerHTML = "";
      live.forEach(([label, col, dash, val]) => {
        const b = document.createElement("b");
        b.innerHTML =
          '<span class="sw"><i style="border-top-color:' + col + ";" + (dash ? "border-top-style:dashed" : "") + '"></i>' + label + "</span>" + '<span class="val">' + Math.round(val) + "</span>";
        legendBox.appendChild(b);
      });
    }
    if (stepRead) stepRead.textContent = ((p * B.xmax) / 1e6).toFixed(2) + "M / " + (B.xmax / 1e6).toFixed(2) + "M steps";
    scrub.value = p;
  }

  const NEXT = (e) => ENVS[(ENVS.indexOf(e) + 1) % ENVS.length];
  function loop(ts) {
    if (!playing) return;
    if (fading) {
      raf = requestAnimationFrame(loop); // hold the animation steady through the cross-fade
      return;
    }
    if (!last) last = ts;
    // rest on the finished curve, then cross-fade to the next task and keep playing
    if (holdUntil) {
      if (ts >= holdUntil) {
        holdUntil = 0;
        crossfadeTo(NEXT(env));
      }
      raf = requestAnimationFrame(loop);
      return;
    }
    p += (ts - last) / 6000;
    last = ts;
    if (p >= 1) {
      p = 1;
      render();
      holdUntil = ts + 900;
      raf = requestAnimationFrame(loop);
      return;
    }
    render();
    raf = requestAnimationFrame(loop);
  }
  playBtn.onclick = () => {
    if (playing) {
      playing = false;
      playBtn.textContent = "play";
      cancelAnimationFrame(raf);
    } else {
      if (p >= 1) p = 0;
      playing = true;
      playBtn.textContent = "pause";
      last = 0;
      holdUntil = 0;
      raf = requestAnimationFrame(loop);
    }
  };
  scrub.addEventListener("input", () => {
    playing = false;
    playBtn.textContent = "play";
    cancelAnimationFrame(raf);
    holdUntil = 0;
    p = +scrub.value;
    render();
  });
  render();
  if (!matchMedia("(prefers-reduced-motion:reduce)").matches) {
    const io = new IntersectionObserver(
      (es) =>
        es.forEach((e) => {
          if (e.isIntersecting && p === 0 && !playing) {
            playBtn.click();
            io.disconnect();
          }
        }),
      { threshold: 0.4 },
    );
    io.observe(plot);
  }
})();
