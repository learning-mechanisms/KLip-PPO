"use strict";
/* experiments page: build the final-return table and the kill-fraction bars */
(function () {
  /* [clip, per-sample, fixed-b, adaptive-b] = [mean, std] */
  var RET = window.RETURNS || [];
  var HARD = new Set(["Ant", "Humanoid"]);
  var body = document.getElementById("retBody");
  if (body) {
    RET.forEach(function (entry) {
      var task = entry[0],
        cols = entry[1];
      var tr = document.createElement("tr");
      if (HARD.has(task)) tr.className = "hard";
      tr.innerHTML =
        "<td>" +
        task +
        "</td>" +
        cols
          .map(function (x) {
            return "<td>" + x[0] + " ± " + x[1] + "</td>";
          })
          .join("");
      body.appendChild(tr);
    });
  }

  /* peak fraction of the PPO-Clip minibatch in I_kill */
  var KILL = window.KILL || [];
  var kb = document.getElementById("killBars");
  if (!kb) return;
  KILL.forEach(function (entry) {
    var task = entry[0],
      frac = entry[1];
    var row = document.createElement("div");
    row.className = "killrow";
    var label = document.createElement("div");
    label.className = "t";
    label.textContent = task;
    var track = document.createElement("div");
    track.className = "killtrack";
    var fill = document.createElement("div");
    fill.className = "killfill";
    fill.dataset.w = (frac / 0.6) * 100;
    track.appendChild(fill);
    var pct = document.createElement("div");
    pct.textContent = (frac * 100).toFixed(1) + "%";
    row.append(label, track, pct);
    kb.appendChild(row);
  });
  new IntersectionObserver(
    function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting)
          kb.querySelectorAll(".killfill").forEach(function (f, i) {
            setTimeout(function () {
              f.style.width = f.dataset.w + "%";
            }, 90 * i);
          });
      });
    },
    { threshold: 0.3 },
  ).observe(kb);
})();
