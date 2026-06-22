"use strict";
/* lightbox.js - click any content figure to open it large in a popup that
   grows out of the clicked thumbnail (a FLIP transform). Close by clicking
   outside the image, the × button, or pressing Escape. */
(function () {
  // content figures only - never the author avatars or inline UI icons
  var SEL = ".gallery img, .twofig figure > img, img.betafig, .algo-fig img";
  var imgs = document.querySelectorAll(SEL);
  if (!imgs.length) return;

  var REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var OPEN_DUR = 300; // ms, grow-in
  var CLOSE_DUR = 220; // ms, shrink back to the thumbnail
  var EASE = "cubic-bezier(.22,.61,.36,1)";

  // overlay, built once and reused
  var lb = document.createElement("div");
  lb.className = "lb";
  lb.setAttribute("role", "dialog");
  lb.setAttribute("aria-modal", "true");
  lb.setAttribute("aria-hidden", "true");

  var closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "lb-close";
  closeBtn.setAttribute("aria-label", "close");
  closeBtn.innerHTML = "&times;";

  var fig = document.createElement("figure");
  fig.className = "lb-fig";
  var big = document.createElement("img");
  big.className = "lb-img";
  big.alt = "";
  var cap = document.createElement("figcaption");
  cap.className = "lb-cap";
  fig.append(big, cap);

  lb.append(closeBtn, fig);
  document.body.appendChild(lb);

  var lastFocus = null;
  var trigger = null;
  var prevOverflow = "";

  // map the final image box onto a target rect: translate centers, scale to size
  function invert(target, from) {
    if (!target.width || !target.height || !from.width || !from.height) return null;
    return (
      "translate(" +
      (from.left + from.width / 2 - (target.left + target.width / 2)) +
      "px," +
      (from.top + from.height / 2 - (target.top + target.height / 2)) +
      "px) scale(" +
      from.width / target.width +
      "," +
      from.height / target.height +
      ")"
    );
  }

  // reuse the page's own caption (cloned, so rendered math comes along)
  function captionFor(img) {
    var f = img.closest("figure");
    if (!f) return null;
    var fc = f.querySelector("figcaption");
    if (!fc || !fc.textContent.trim()) return null;
    var node = fc.cloneNode(true);
    node.removeAttribute("class"); // shed page styling; .lb-cap handles it
    return node;
  }

  function open(img) {
    lastFocus = document.activeElement;
    trigger = img;
    big.src = img.currentSrc || img.src;
    big.alt = img.alt || "";
    lb.setAttribute("aria-label", img.alt || "figure");
    cap.innerHTML = "";
    var c = captionFor(img);
    if (c) {
      cap.appendChild(c);
      cap.style.display = "";
    } else {
      cap.style.display = "none";
    }

    lb.setAttribute("aria-hidden", "false");
    prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    lb.classList.add("open"); // backdrop + caption fade in
    closeBtn.focus();

    big.style.transition = "none";
    big.style.transform = "none";
    if (REDUCE) {
      big.style.opacity = "1";
      return;
    }

    var from = img.getBoundingClientRect();
    big.style.opacity = "0"; // hide until inverted, so there's no flash at full size

    var run = function () {
      if (!lb.classList.contains("open")) return; // closed during decode
      var t = invert(big.getBoundingClientRect(), from);
      big.style.transition = "none";
      big.style.transform = t || "none";
      void big.offsetWidth; // commit the start state
      big.style.transition = "transform " + OPEN_DUR + "ms " + EASE + ", opacity " + Math.round(OPEN_DUR * 0.6) + "ms ease";
      big.style.transform = "none";
      big.style.opacity = "1";
    };

    // the thumbnail is already cached, so this resolves almost immediately
    if (big.complete && big.naturalWidth) requestAnimationFrame(run);
    else if (big.decode) big.decode().then(run, run);
    else big.onload = run;
  }

  function finishClose() {
    lb.setAttribute("aria-hidden", "true");
    document.body.style.overflow = prevOverflow;
    // leave the image shrunk/hidden while the overlay is closed; open() resets it
    big.style.transition = "none";
  }

  function close() {
    if (!lb.classList.contains("open")) return;
    lb.classList.remove("open"); // backdrop + caption fade out

    var to = trigger && trigger.getBoundingClientRect();
    var from = big.getBoundingClientRect();
    var t = REDUCE ? null : to && invert(from, to);
    if (!t) {
      finishClose();
      if (lastFocus && lastFocus.focus) lastFocus.focus();
      return;
    }

    var done = false;
    function finish(e) {
      if (done || (e && e.propertyName && e.propertyName !== "transform")) return;
      done = true;
      big.removeEventListener("transitionend", finish);
      finishClose();
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }
    big.style.transition = "transform " + CLOSE_DUR + "ms cubic-bezier(.4,0,.2,1), opacity " + CLOSE_DUR + "ms ease";
    void big.offsetWidth;
    big.style.transform = t; // shrink back toward the thumbnail
    big.style.opacity = "0";
    big.addEventListener("transitionend", finish);
    setTimeout(finish, CLOSE_DUR + 80); // fallback if transitionend is missed
  }

  imgs.forEach(function (img) {
    img.classList.add("zoomable");
    img.setAttribute("role", "button");
    img.setAttribute("tabindex", "0");
    if (!img.getAttribute("aria-label")) img.setAttribute("aria-label", (img.alt ? img.alt + " - " : "") + "view larger");
    img.addEventListener("click", function () {
      open(img);
    });
    img.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open(img);
      }
    });
  });

  // click on the backdrop (or the figure's empty margin) closes; clicks on the image do not
  lb.addEventListener("click", function (e) {
    if (e.target === lb || e.target === fig) close();
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });
})();
