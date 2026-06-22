"use strict";
/* overview page: the three-forms accordion and the flowchart selector */
(function () {
  document.querySelectorAll("#forms .row > button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var row = btn.parentElement,
        wasOpen = row.classList.contains("open");
      document.querySelectorAll("#forms .row").forEach(function (r) {
        r.classList.remove("open");
      });
      if (!wasOpen) row.classList.add("open");
    });
  });

  document.querySelectorAll(".algo-tabs .algo-tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      var pick = tab.dataset.algo;
      document.querySelectorAll(".algo-tabs .algo-tab").forEach(function (t) {
        var on = t.dataset.algo === pick;
        t.classList.toggle("on", on);
        t.setAttribute("aria-selected", on);
      });
      document.querySelectorAll(".algo-fig").forEach(function (f) {
        f.hidden = f.dataset.algo !== pick;
      });
    });
  });
})();
