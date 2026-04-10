(() => {
  const INPUT_STATE_KEY = "autofigure_input_state_v3";

  const page = document.body.dataset.page;
  if (page === "input") {
    initInputPage();
  } else if (page === "canvas") {
    initCanvasPage();
  }

  function $(id) {
    return document.getElementById(id);
  }

  function initInputPage() {
    const confirmBtn = $("confirmBtn");
    const errorMsg = $("errorMsg");
    const uploadZone = $("uploadZone");
    const referenceFile = $("referenceFile");
    const referencePreview = $("referencePreview");
    const referenceStatus = $("referenceStatus");
    const imageSizeGroup = $("imageSizeGroup");
    const imageSizeInput = $("imageSize");
    const samBackend = $("samBackend");
    const samPrompt = $("samPrompt");
    const samApiKeyGroup = $("samApiKeyGroup");
    const samApiKeyInput = $("samApiKey");
    let uploadedReferencePath = null;

    function loadInputState() {
      try {
        const raw = window.sessionStorage.getItem(INPUT_STATE_KEY);
        if (!raw) {
          return null;
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : null;
      } catch (_err) {
        return null;
      }
    }

    function saveInputState() {
      const state = {
        methodText: $("methodText")?.value ?? "",
        provider: $("provider")?.value ?? "gemini",
        apiKey: $("apiKey")?.value ?? "",
        optimizeIterations: $("optimizeIterations")?.value ?? "0",
        imageSize: imageSizeInput?.value ?? "4K",
        samBackend: samBackend?.value ?? "roboflow",
        samPrompt: samPrompt?.value ?? "icon,person,robot,animal,arrow,diagram,frame,connector",
        samApiKey: samApiKeyInput?.value ?? "",
        referencePath: uploadedReferencePath,
        referenceUrl: referencePreview?.src ?? "",
        referenceStatus: referenceStatus?.textContent ?? "",
      };
      try {
        window.sessionStorage.setItem(INPUT_STATE_KEY, JSON.stringify(state));
      } catch (_err) {
        // Ignore storage failures (e.g. private mode / quota)
      }
    }

    function applyInputState() {
      const state = loadInputState();
      if (!state) {
        return;
      }
      if (typeof state.methodText === "string") {
        $("methodText").value = state.methodText;
      }
      if (typeof state.provider === "string" && $("provider")) {
        $("provider").value = state.provider;
      }
      if (typeof state.apiKey === "string") {
        $("apiKey").value = state.apiKey;
      }
      if (typeof state.optimizeIterations === "string" && $("optimizeIterations")) {
        $("optimizeIterations").value = state.optimizeIterations;
      }
      if (typeof state.imageSize === "string" && imageSizeInput) {
        imageSizeInput.value = state.imageSize;
      }
      if (typeof state.samBackend === "string" && samBackend) {
        samBackend.value = state.samBackend;
      }
      if (typeof state.samPrompt === "string" && samPrompt) {
        samPrompt.value = state.samPrompt;
      }
      if (typeof state.samApiKey === "string" && samApiKeyInput) {
        samApiKeyInput.value = state.samApiKey;
      }
      if (typeof state.referencePath === "string" && state.referencePath) {
        uploadedReferencePath = state.referencePath;
      }
      if (
        referencePreview &&
        typeof state.referenceUrl === "string" &&
        state.referenceUrl
      ) {
        referencePreview.src = state.referenceUrl;
        referencePreview.classList.add("visible");
      }
      if (
        referenceStatus &&
        typeof state.referenceStatus === "string" &&
        state.referenceStatus
      ) {
        referenceStatus.textContent = state.referenceStatus;
      }
    }

    let _serverDefaults = null;

    function syncApiKeyForProvider() {
      if (!_serverDefaults || !$("apiKey")) return;
      const provider = $("provider")?.value ?? "gemini";
      const hasKey = provider === "gemini" ? _serverDefaults.hasGoogleApiKey : _serverDefaults.hasApiKey;
      if (hasKey) {
        $("apiKey").placeholder = "(server key configured)";
        $("apiKey").value = "";
      }
      saveInputState();
    }

    function syncImageSizeVisibility() {
      const provider = $("provider")?.value ?? "gemini";
      const show = provider === "gemini";
      if (imageSizeGroup) {
        imageSizeGroup.hidden = !show;
      }
      saveInputState();
    }

    function syncSamApiKeyVisibility() {
      const shouldShow =
        samBackend &&
        (samBackend.value === "fal" || samBackend.value === "roboflow");
      if (samApiKeyGroup) {
        samApiKeyGroup.hidden = !shouldShow;
      }
      if (!shouldShow && samApiKeyInput) {
        samApiKeyInput.value = "";
      }
      saveInputState();
    }

    applyInputState();

    // Pre-fill from server defaults for any empty key fields
    (async () => {
      try {
        const res = await fetch("/api/defaults");
        if (res.ok) {
          const defs = await res.json();
          _serverDefaults = defs;
          if ($("apiKey")) {
            const provider = $("provider")?.value ?? "gemini";
            const hasKey = provider === "gemini" ? defs.hasGoogleApiKey : defs.hasApiKey;
            if (hasKey) {
              $("apiKey").placeholder = "(server key configured)";
              $("apiKey").value = "";
            }
          }
          if (defs.hasSamApiKey && samApiKeyInput && !samApiKeyInput.value) {
            samApiKeyInput.placeholder = "(server key configured)";
            samApiKeyInput.value = "";
          }
          saveInputState();
        }
      } catch (_err) { /* silently ignore */ }
    })();

    if (samBackend) {
      samBackend.addEventListener("change", syncSamApiKeyVisibility);
      syncSamApiKeyVisibility();
    }
    if ($("provider")) {
      $("provider").addEventListener("change", () => {
        syncImageSizeVisibility();
        syncApiKeyForProvider();
      });
      syncImageSizeVisibility();
    }

    if (uploadZone && referenceFile) {
      uploadZone.addEventListener("click", () => referenceFile.click());
      uploadZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadZone.classList.add("dragging");
      });
      uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragging");
      });
      uploadZone.addEventListener("drop", async (event) => {
        event.preventDefault();
        uploadZone.classList.remove("dragging");
        const file = event.dataTransfer.files[0];
        if (file) {
          const uploadedRef = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedRef) {
            uploadedReferencePath = uploadedRef.path;
            saveInputState();
          }
        }
      });
      referenceFile.addEventListener("change", async () => {
        const file = referenceFile.files[0];
        if (file) {
          const uploadedRef = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedRef) {
            uploadedReferencePath = uploadedRef.path;
            saveInputState();
          }
        }
      });
    }

    const autoSaveFields = [
      $("methodText"),
      $("provider"),
      $("apiKey"),
      $("optimizeIterations"),
      $("imageSize"),
      samPrompt,
      samApiKeyInput,
    ];
    for (const field of autoSaveFields) {
      if (!field) {
        continue;
      }
      field.addEventListener("input", saveInputState);
      field.addEventListener("change", saveInputState);
    }

    confirmBtn.addEventListener("click", async () => {
      errorMsg.textContent = "";
      const methodText = $("methodText").value.trim();
      if (!methodText) {
        errorMsg.textContent = "Please provide method text.";
        return;
      }

      confirmBtn.disabled = true;
      confirmBtn.textContent = "Starting...";

      const payload = {
        method_text: methodText,
        provider: $("provider").value,
        api_key: $("apiKey").value.trim() || null,
        optimize_iterations: parseInt($("optimizeIterations").value, 10),
        reference_image_path: uploadedReferencePath,
        sam_backend: $("samBackend").value,
        sam_prompt: $("samPrompt").value.trim() || null,
        sam_api_key: $("samApiKey").value.trim() || null,
      };
      if ($("provider").value === "gemini") {
        payload.image_size = imageSizeInput?.value || "4K";
      }
      if (payload.sam_backend === "local") {
        payload.sam_api_key = null;
      }
      saveInputState();

      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "Request failed");
        }

        const data = await response.json();
        window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}`;
      } catch (err) {
        errorMsg.textContent = err.message || "Failed to start job";
        confirmBtn.disabled = false;
        confirmBtn.textContent = "Confirm -> Canvas";
      }
    });

    initSessionsExplorer();
  }

  async function initSessionsExplorer() {
    const grid = document.getElementById("sessionsGrid");
    const empty = document.getElementById("sessionsEmpty");
    const countEl = document.getElementById("sessionsCount");
    if (!grid) return;

    let sessions;
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) return;
      sessions = await res.json();
    } catch (_err) {
      return;
    }

    if (!sessions.length) return;

    if (empty) empty.remove();
    if (countEl) countEl.textContent = `${sessions.length} run${sessions.length === 1 ? "" : "s"}`;

    for (const session of sessions) {
      grid.appendChild(buildSessionCard(session, grid, countEl));
    }
  }

  function buildSessionCard(session, grid, countEl) {
    const card = document.createElement("div");
    card.className = "session-card";

    // thumbnail
    const thumb = document.createElement("div");
    thumb.className = "session-thumb";
    if (session.figure_url) {
      const img = document.createElement("img");
      img.src = session.figure_url;
      img.className = "session-img";
      img.loading = "lazy";
      img.alt = "";
      thumb.appendChild(img);
      thumb.classList.add("session-thumb--zoomable");
      thumb.addEventListener("click", () => openLightbox(session.figure_url));
    } else {
      thumb.innerHTML = `<svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 15l4-4 3 3 4-5 7 7"/></svg>`;
    }

    // meta
    const meta = document.createElement("div");
    meta.className = "session-meta";

    const topRow = document.createElement("div");
    topRow.className = "session-meta-top";

    const date = document.createElement("div");
    date.className = "session-date";
    date.textContent = formatSessionDate(session.created_at);

    if (session.has_final_svg) {
      const badge = document.createElement("span");
      badge.className = "session-badge session-badge--done";
      badge.textContent = "Done";
      topRow.appendChild(badge);
    }
    topRow.appendChild(date);

    const actions = document.createElement("div");
    actions.className = "session-actions";

    const openBtn = document.createElement("button");
    openBtn.className = "session-open-btn";
    openBtn.textContent = "Open";
    openBtn.addEventListener("click", async () => {
      openBtn.disabled = true;
      openBtn.textContent = "…";
      try {
        const res = await fetch(`/api/sessions/${session.job_id}/open`, { method: "POST" });
        if (!res.ok) throw new Error();
        const data = await res.json();
        window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}`;
      } catch (_err) {
        openBtn.disabled = false;
        openBtn.textContent = "Open";
      }
    });

    const delBtn = document.createElement("button");
    delBtn.className = "session-del-btn";
    delBtn.title = "Delete this run";
    delBtn.innerHTML = `<svg viewBox="0 0 20 20" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polyline points="5 7 5 16 15 16 15 7"/><line x1="3" y1="7" x2="17" y2="7"/><polyline points="8 7 8 4 12 4 12 7"/></svg>`;
    delBtn.addEventListener("click", () => {
      showDeleteConfirm(session.job_id, session.created_at, card, grid, countEl);
    });

    actions.appendChild(openBtn);
    actions.appendChild(delBtn);
    meta.appendChild(topRow);
    meta.appendChild(actions);
    card.appendChild(thumb);
    card.appendChild(meta);
    return card;
  }

  function showDeleteConfirm(jobId, createdAt, card, grid, countEl) {
    // remove any existing modal
    document.getElementById("deleteModal")?.remove();

    const overlay = document.createElement("div");
    overlay.id = "deleteModal";
    overlay.className = "modal-overlay";

    const box = document.createElement("div");
    box.className = "modal-box";

    const title = document.createElement("div");
    title.className = "modal-title";
    title.textContent = "Supprimer ce run ?";

    const sub = document.createElement("div");
    sub.className = "modal-sub";
    sub.textContent = `${formatSessionDate(createdAt)} — cette action est irréversible.`;

    const btnRow = document.createElement("div");
    btnRow.className = "modal-actions";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "modal-cancel";
    cancelBtn.textContent = "Annuler";
    cancelBtn.addEventListener("click", () => overlay.remove());

    const confirmBtn = document.createElement("button");
    confirmBtn.className = "modal-confirm";
    confirmBtn.textContent = "Supprimer";
    confirmBtn.addEventListener("click", async () => {
      confirmBtn.disabled = true;
      confirmBtn.textContent = "…";
      try {
        const res = await fetch(`/api/sessions/${jobId}`, { method: "DELETE" });
        if (!res.ok) throw new Error();
        overlay.remove();
        card.classList.add("session-card--removing");
        card.addEventListener("transitionend", () => {
          card.remove();
          // update count
          const remaining = grid.querySelectorAll(".session-card").length;
          if (countEl) countEl.textContent = remaining ? `${remaining} run${remaining === 1 ? "" : "s"}` : "";
          if (!remaining) {
            const empty = document.createElement("div");
            empty.className = "sessions-empty";
            empty.textContent = "No previous runs yet.";
            grid.appendChild(empty);
          }
        }, { once: true });
      } catch (_err) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = "Supprimer";
      }
    });

    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(confirmBtn);
    box.appendChild(title);
    box.appendChild(sub);
    box.appendChild(btnRow);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add("modal-overlay--in"));
  }

  function openLightbox(url) {
    document.getElementById("lightboxOverlay")?.remove();

    const overlay = document.createElement("div");
    overlay.id = "lightboxOverlay";
    overlay.className = "lightbox-overlay";

    const img = document.createElement("img");
    img.src = url;
    img.className = "lightbox-img";
    img.alt = "";

    const closeBtn = document.createElement("button");
    closeBtn.className = "lightbox-close";
    closeBtn.innerHTML = `<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="3" x2="17" y2="17"/><line x1="17" y1="3" x2="3" y2="17"/></svg>`;
    closeBtn.addEventListener("click", () => closeLightbox(overlay));

    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeLightbox(overlay); });
    document.addEventListener("keydown", function onKey(e) {
      if (e.key === "Escape") { closeLightbox(overlay); document.removeEventListener("keydown", onKey); }
    });

    overlay.appendChild(img);
    overlay.appendChild(closeBtn);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add("lightbox-overlay--in"));
  }

  function closeLightbox(overlay) {
    overlay.classList.remove("lightbox-overlay--in");
    overlay.addEventListener("transitionend", () => overlay.remove(), { once: true });
  }

  function formatSessionDate(isoStr) {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  }

  async function uploadReference(file, confirmBtn, previewEl, statusEl) {
    if (!file.type.startsWith("image/")) {
      statusEl.textContent = "Only image files are supported.";
      return null;
    }

    confirmBtn.disabled = true;
    statusEl.textContent = "Uploading reference...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Upload failed");
      }

      const data = await response.json();
      statusEl.textContent = `Using uploaded reference: ${data.name}`;
      if (previewEl) {
        previewEl.src = data.url || "";
        previewEl.classList.add("visible");
      }
      return {
        path: data.path || null,
        url: data.url || "",
        name: data.name || "",
      };
    } catch (err) {
      statusEl.textContent = err.message || "Upload failed";
      return null;
    } finally {
      confirmBtn.disabled = false;
    }
  }

  async function initCanvasPage() {
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job");
    const statusText = $("statusText");
    const jobIdEl = $("jobId");
    const artifactPanel = $("artifactPanel");
    const artifactList = $("artifactList");
    const toggle = $("artifactToggle");
    const logToggle = $("logToggle");
    const backToConfigBtn = $("backToConfigBtn");
    const logPanel = $("logPanel");
    const logBody = $("logBody");
    const downloadBtn = $("downloadBtn");
    const iframe = $("svgEditorFrame");
    const fallback = $("svgFallback");
    const fallbackObject = $("fallbackObject");
    const saveStatusEl = $("saveStatus");

    let lastSavedSvg = null;
    let isSaving = false;
    let autoSaveIntervalId = null;

    function getSvgStringFromEditor() {
      const win = iframe.contentWindow;
      if (!win) return null;
      try {
        if (win.svgCanvas && typeof win.svgCanvas.getSvgString === "function") {
          return win.svgCanvas.getSvgString();
        }
        if (win.svgEditor && win.svgEditor.svgCanvas && typeof win.svgEditor.svgCanvas.getSvgString === "function") {
          return win.svgEditor.svgCanvas.getSvgString();
        }
      } catch (_) {
        // cross-origin or not yet ready
      }
      return null;
    }

    function startAutoSave() {
      if (autoSaveIntervalId) return;
      autoSaveIntervalId = setInterval(autoSaveSvg, 3000);
    }

    async function autoSaveSvg() {
      if (isSaving) return;
      const svgText = getSvgStringFromEditor();
      if (!svgText || svgText === lastSavedSvg) return;
      isSaving = true;
      if (saveStatusEl) saveStatusEl.textContent = "Saving\u2026";
      try {
        const res = await fetch(`/api/sessions/${jobId}/svg`, {
          method: "PUT",
          headers: { "Content-Type": "image/svg+xml" },
          body: svgText,
        });
        if (res.ok) {
          lastSavedSvg = svgText;
          if (saveStatusEl) {
            saveStatusEl.textContent = "Saved";
            setTimeout(() => {
              if (saveStatusEl.textContent === "Saved") saveStatusEl.textContent = "";
            }, 2000);
          }
        } else {
          if (saveStatusEl) saveStatusEl.textContent = "Save failed";
        }
      } catch (_) {
        if (saveStatusEl) saveStatusEl.textContent = "Save failed";
      } finally {
        isSaving = false;
      }
    }

    if (!jobId) {
      statusText.textContent = "Missing job id";
      return;
    }

    jobIdEl.textContent = jobId;
    if (downloadBtn) {
      downloadBtn.style.display = "";
      downloadBtn.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = `/api/sessions/${jobId}/download`;
        a.download = `${jobId}.zip`;
        a.click();
      });
    }

    toggle.addEventListener("click", () => {
      artifactPanel.classList.toggle("open");
    });

    logToggle.addEventListener("click", () => {
      logPanel.classList.toggle("open");
    });
    if (backToConfigBtn) {
      backToConfigBtn.addEventListener("click", () => {
        window.location.href = "/";
      });
    }

    let svgEditAvailable = false;
    let svgEditPath = null;
    try {
      const configRes = await fetch("/api/config");
      if (configRes.ok) {
        const config = await configRes.json();
        svgEditAvailable = Boolean(config.svgEditAvailable);
        svgEditPath = config.svgEditPath || null;
      }
    } catch (err) {
      svgEditAvailable = false;
    }

    if (svgEditAvailable && svgEditPath) {
      iframe.src = svgEditPath;
    } else {
      fallback.classList.add("active");
      iframe.style.display = "none";
    }

    let svgReady = false;
    let pendingSvgText = null;

    iframe.addEventListener("load", () => {
      svgReady = true;
      if (pendingSvgText) {
        const text = pendingSvgText;
        pendingSvgText = null;
        const loaded = tryLoadSvg(text);
        if (loaded) {
          lastSavedSvg = text;
          startAutoSave();
        }
      }
    });

    const stepMap = {
      figure: { step: 1, label: "Figure generated" },
      samed: { step: 2, label: "SAM3 segmentation" },
      icon_raw: { step: 3, label: "Icons extracted" },
      icon_nobg: { step: 3, label: "Icons refined" },
      template_svg: { step: 4, label: "Template SVG ready" },
      final_svg: { step: 5, label: "Final SVG ready" },
    };

    let currentStep = 0;

    const artifacts = new Set();
    const eventSource = new EventSource(`/api/events/${jobId}`);
    let isFinished = false;

    let figureUrl = null;
    const figureFab = $("figureFab");
    if (figureFab) {
      figureFab.addEventListener("click", () => {
        if (figureUrl) openLightbox(figureUrl);
      });
    }

    eventSource.addEventListener("artifact", async (event) => {
      const data = JSON.parse(event.data);
      if (!artifacts.has(data.path)) {
        artifacts.add(data.path);
        addArtifactCard(artifactList, data);
      }

      if (data.kind === "figure") {
        figureUrl = data.url;
        if (figureFab) figureFab.classList.remove("is-hidden");
      }

      if (data.kind === "template_svg" || data.kind === "final_svg") {
        await loadSvgAsset(data.url);
      }

      if (stepMap[data.kind] && stepMap[data.kind].step > currentStep) {
        currentStep = stepMap[data.kind].step;
        statusText.textContent = `Step ${currentStep}/5 - ${stepMap[data.kind].label}`;
      }
    });

    eventSource.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.state === "started") {
        statusText.textContent = "Running";
      } else if (data.state === "finished") {
        isFinished = true;
        if (typeof data.code === "number" && data.code !== 0) {
          statusText.textContent = `Failed (code ${data.code})`;
        } else {
          statusText.textContent = "Done";
        }
      }
    });

    eventSource.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      appendLogLine(logBody, data);
    });

    eventSource.onerror = () => {
      if (isFinished) {
        eventSource.close();
        return;
      }
      statusText.textContent = "Disconnected";
    };

    async function loadSvgAsset(url) {
      let svgText = "";
      try {
        const response = await fetch(url);
        svgText = await response.text();
      } catch (err) {
        return;
      }

      if (svgEditAvailable) {
        if (!svgEditPath) {
          return;
        }
        if (!svgReady) {
          pendingSvgText = svgText;
          return;
        }

        const loaded = tryLoadSvg(svgText);
        if (loaded) {
          lastSavedSvg = svgText;
          startAutoSave();
        } else {
          iframe.src = `${svgEditPath}?url=${encodeURIComponent(url)}`;
        }
      } else {
        fallbackObject.data = url;
      }
    }

    function tryLoadSvg(svgText) {
      if (!iframe.contentWindow) {
        return false;
      }

      const win = iframe.contentWindow;
      if (win.svgEditor && typeof win.svgEditor.loadFromString === "function") {
        win.svgEditor.loadFromString(svgText);
        return true;
      }
      if (win.svgCanvas && typeof win.svgCanvas.setSvgString === "function") {
        win.svgCanvas.setSvgString(svgText);
        return true;
      }
      return false;
    }
  }

  function appendLogLine(container, data) {
    const line = `[${data.stream}] ${data.line}`;
    const lines = container.textContent.split("\n").filter(Boolean);
    lines.push(line);
    if (lines.length > 200) {
      lines.splice(0, lines.length - 200);
    }
    container.textContent = lines.join("\n");
    container.scrollTop = container.scrollHeight;
  }

  function addArtifactCard(container, data) {
    const card = document.createElement("a");
    card.className = "artifact-card";
    card.href = data.url;
    card.target = "_blank";
    card.rel = "noreferrer";

    const img = document.createElement("img");
    img.src = data.url;
    img.alt = data.name;
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "artifact-meta";

    const name = document.createElement("div");
    name.className = "artifact-name";
    name.textContent = data.name;

    const badge = document.createElement("div");
    badge.className = "artifact-badge";
    badge.textContent = formatKind(data.kind);

    meta.appendChild(name);
    meta.appendChild(badge);
    card.appendChild(img);
    card.appendChild(meta);
    container.prepend(card);
  }

  function formatKind(kind) {
    switch (kind) {
      case "figure":
        return "figure";
      case "samed":
        return "samed";
      case "icon_raw":
        return "icon raw";
      case "icon_nobg":
        return "icon no-bg";
      case "template_svg":
        return "template";
      case "final_svg":
        return "final";
      default:
        return "artifact";
    }
  }
})();
