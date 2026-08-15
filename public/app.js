// app.js (updated) ---------------------------------------------
// NOTE: This file extends your previous app.js with "My Schedule & Tasks"
// frontend logic. It assumes the schedule API endpoints exist as implemented
// on the backend: /api/schedule/<user_id> and related routes.
//
// Keep this file as a single drop-in replacement for your existing app.js.
// ------------------------------------------------------------------

// ------- simple API helper --------
const API_BASE = "";

async function api(path, options = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || data.message || "Request failed");
  return data;
}

// ------- state -------
let currentUser = null;
let dashboardData = null;
let allCoursesCache = [];
let scheduleCache = []; // cache tasks for current user
let notificationTimers = []; // scheduled notification timeouts

// ================= Focus Detection (ADDED) =================
let focusStream = null;
let focusInterval = null;
let focusCameraVideo = null;
let cameraPreviewEl = null;
let cameraEnabled = false;
let currentVideoEl = null;
let currentFocusStatusEl = null;
let focusBlocked = false;

// ===== UI helpers =====
function humanizeFilename(name) {
  if (!name) return "";
  return name
    .replace(/_/g, " ")
    .replace(/\.(webm|mp4|mkv)$/i, "")
    .trim();
}

// ===== YouTube embed helpers =====
// Course videos may point at a YouTube URL (unlisted uploads) instead of a
// locally-hosted file -- large video files served directly from a home
// connection's upload bandwidth don't play reliably for remote viewers, so
// some/all courses may use YouTube for hosting instead.
function extractYouTubeId(url) {
  if (!url) return null;
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtube\.com\/embed\/|youtu\.be\/)([a-zA-Z0-9_-]{11})/,
  ];
  for (const re of patterns) {
    const m = url.match(re);
    if (m) return m[1];
  }
  return null;
}

let _youTubeApiPromise = null;
function loadYouTubeApi() {
  if (_youTubeApiPromise) return _youTubeApiPromise;
  _youTubeApiPromise = new Promise((resolve) => {
    if (window.YT && window.YT.Player) {
      resolve(window.YT);
      return;
    }
    const prevCallback = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      if (typeof prevCallback === "function") prevCallback();
      resolve(window.YT);
    };
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  });
  return _youTubeApiPromise;
}


// ==========================================================

// ------- DOM refs -------
// Auth
const authContainer = document.getElementById("auth-container");
const appContainer = document.getElementById("app-container");

const loginForm = document.getElementById("login-form");
const signupForm = document.getElementById("signup-form");
const loginTab = document.getElementById("login-tab");
const signupTab = document.getElementById("signup-tab");

const loginError = document.getElementById("login-error");
const loginSuccess = document.getElementById("login-success");
const signupError = document.getElementById("signup-error");
const signupSuccess = document.getElementById("signup-success");

const forgotLink = document.getElementById("forgot-link");
const forgotModal = document.getElementById("forgot-modal");
const forgotEmail = document.getElementById("forgot-email");
const forgotSubmit = document.getElementById("forgot-submit");
const forgotClose = document.getElementById("forgot-close");
const forgotMessage = document.getElementById("forgot-message");

// Navbar / nav
const logoutBtn = document.getElementById("logout-btn");
const navProfileBtn = document.getElementById("nav-profile-btn");
const navRoutes = document.querySelectorAll(".topnav-route");
const sideLinks = document.querySelectorAll(".side-link[data-view]");
const views = document.querySelectorAll(".view");
const avatarInitial = document.getElementById("nav-avatar-initial");

// Start learning mega menu
const startLearningBtn = document.getElementById("start-learning-btn");
const startLearningPanel = document.getElementById("start-learning-panel");

// Dashboard
const coursesOverviewEl = document.getElementById("courses-overview");
const overallDonut = document.getElementById("overall-donut");
const overallPercentEl = document.getElementById("overall-percent");
const weeklyHoursEl = document.getElementById("weekly-hours");
const statusTextEl = document.getElementById("status-text");
const recentBarsEl = document.getElementById("recent-bars");

const heatmapMonthLabelsEl = document.getElementById("heatmap-month-labels");
const heatmapGridEl = document.getElementById("heatmap-grid");

const todoListEl = document.getElementById("todo-list");
const saveTodosBtn = document.getElementById("save-todos");
const recommendedEl = document.getElementById("recommended");
const announcementsEl = document.getElementById("announcements");

// Hero stats
const heroWeekHoursEl = document.getElementById("hero-week-hours");
const heroCoursesProgressEl = document.getElementById("hero-courses-progress");
const heroSearchBtn = document.getElementById("hero-search-btn");
const heroDomainSelect = document.getElementById("hero-domain-select");

// Courses view
const coursesListEl = document.getElementById("courses-list");
const courseDetailTitle = document.getElementById("course-detail-title");
const courseDetailDesc = document.getElementById("course-detail-description");
const courseDetailMeta = document.getElementById("course-detail-meta");
const courseModulesEl = document.getElementById("course-modules");

// Profile
const profileNameEl = document.getElementById("profile-name");
const profileEmailEl = document.getElementById("profile-email");
const profileRoleEl = document.getElementById("profile-role");
const profileInterestsEl = document.getElementById("profile-interests");

// ---- Schedule & Tasks DOM refs (may be present in dashboard or a dedicated view) ----
const scheduleViewEl = document.getElementById("schedule-view");
const todayTasksEl = document.getElementById("today-tasks");
const weekStripEl = document.getElementById("week-strip");
const addTaskBtn = document.getElementById("add-task-btn");
const taskModal = document.getElementById("task-modal");
const taskModalTitle = document.getElementById("task-modal-title");
const taskForm = document.getElementById("task-form");
const taskFormTitle = document.getElementById("task-title");
const taskFormNotes = document.getElementById("task-notes");
const taskFormDate = document.getElementById("task-due-date");
const taskFormTime = document.getElementById("task-due-time");
const taskFormDuration = document.getElementById("task-duration");
const taskFormPriority = document.getElementById("task-priority");
const taskFormRecurrence = document.getElementById("task-recurrence");
const taskFormSave = document.getElementById("task-save");
const taskFormCancel = document.getElementById("task-cancel");
const taskModalMsg = document.getElementById("task-modal-msg");

// ------- auth tab switching -------
if (loginTab && signupTab && loginForm && signupForm) {
  loginTab.addEventListener("click", () => {
    loginTab.classList.add("active");
    signupTab.classList.remove("active");
    loginForm.classList.remove("hidden");
    signupForm.classList.add("hidden");
  });

  signupTab.addEventListener("click", () => {
    signupTab.classList.add("active");
    loginTab.classList.remove("active");
    signupForm.classList.remove("hidden");
    loginForm.classList.add("hidden");
  });
}

// ------- forgot modal code (left as-is) -------
if (forgotLink) {
  forgotLink.addEventListener("click", (e) => {
    e.preventDefault();
    if (forgotModal) forgotModal.classList.remove("hidden");
    if (forgotEmail) forgotEmail.value = "";
    if (forgotMessage) forgotMessage.textContent = "";
  });
}
if (forgotClose) forgotClose.addEventListener("click", () => forgotModal && forgotModal.classList.add("hidden"));
if (forgotSubmit) {
  forgotSubmit.addEventListener("click", async () => {
    try {
      const email = forgotEmail.value.trim();
      if (!email) {
        forgotMessage.textContent = "Please enter your email.";
        return;
      }
      const resp = await api("/api/auth/forgot", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      forgotMessage.textContent = resp.message;
    } catch (err) {
      forgotMessage.textContent = err.message;
    }
  });
}

// ------- signup/login/logout (unchanged) -------
if (signupForm) {
  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (signupError) signupError.textContent = "";
    if (signupSuccess) signupSuccess.textContent = "";

    const name = document.getElementById("signup-name").value.trim();
    const email = document.getElementById("signup-email").value.trim();
    const password = document.getElementById("signup-password").value;
    const confirm = document.getElementById("signup-confirm").value;

    // Role is fixed to student
    const role = "student";

    if (password !== confirm) {
      signupError.textContent = "Passwords do not match.";
      return;
    }

    try {
      const resp = await api("/api/auth/signup", {
        method: "POST",
        body: JSON.stringify({
          name,
          email,
          password,
          role
        }),
      });

      signupSuccess.textContent = "Account created successfully. Please log in.";
      signupForm.reset();

      // switch to login tab
      loginTab && loginTab.click();

    } catch (err) {
      signupError.textContent = err.message || "Signup failed.";
    }
  });
}


if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (loginError) loginError.textContent = "";
    if (loginSuccess) loginSuccess.textContent = "";
    const email = document.getElementById("login-email").value.trim();
    const password = document.getElementById("login-password").value;
    if (!email || !password) {
      loginError.textContent = "Please fill in both fields.";
      return;
    }
    try {
      const resp = await api("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      currentUser = resp.user;
      localStorage.setItem("lmsUser", JSON.stringify(currentUser));
      await loadDashboard();
      authContainer && authContainer.classList.add("hidden");
      appContainer && appContainer.classList.remove("hidden");
    } catch (err) {
      loginError.textContent = err.message;
    }
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch (e) {
      // ignore -- still proceed to clear local state even if the
      // backend call fails (e.g. offline), so the user isn't stuck
    }
    currentUser = null;
    dashboardData = null;
    localStorage.removeItem("lmsUser");
    appContainer && appContainer.classList.add("hidden");
    authContainer && authContainer.classList.remove("hidden");
  });
}

// navigation and mega menu behavior (unchanged)
function setActiveView(viewId) {
  views.forEach((v) => v.classList.remove("active"));
  const el = document.getElementById(viewId);
  if (el) el.classList.add("active");
  sideLinks.forEach((btn) =>
    btn.dataset.view === viewId ? btn.classList.add("active") : btn.classList.remove("active")
  );
}

navRoutes.forEach((btn) => {
  btn.addEventListener("click", () => {
    const viewId = btn.dataset.view;
    if (viewId) {
      setActiveView(viewId);
      const sideMatch = Array.from(sideLinks).find((s) => s.dataset.view === viewId);
      if (sideMatch) {
        sideLinks.forEach((s) => s.classList.remove("active"));
        sideMatch.classList.add("active");
      }
      if (viewId === "courses-view") loadCourses();
      if (viewId === "schedule-view") loadScheduleForCurrentUser();
    }
  });
});
sideLinks.forEach((btn) => {
  btn.addEventListener("click", () => {
    const viewId = btn.dataset.view;
    setActiveView(viewId);
    if (viewId === "courses-view") loadCourses();
    if (viewId === "schedule-view") loadScheduleForCurrentUser();
  });
});

// start learning mega menu handling (unchanged)
let megaOpen = false;
function openMega() { startLearningPanel && startLearningPanel.classList.remove("hidden"); startLearningBtn && startLearningBtn.setAttribute("aria-expanded", "true"); megaOpen = true; }
function closeMega(){ startLearningPanel && startLearningPanel.classList.add("hidden"); startLearningBtn && startLearningBtn.setAttribute("aria-expanded","false"); megaOpen = false; }
if (startLearningBtn) startLearningBtn.addEventListener("click", (e)=> { e.stopPropagation(); megaOpen ? closeMega() : openMega(); });
if (startLearningBtn) startLearningBtn.addEventListener("mouseenter", ()=> openMega());
if (startLearningPanel) startLearningPanel.addEventListener("mouseenter", ()=> openMega());
if (startLearningPanel) startLearningPanel.addEventListener("mouseleave", ()=> closeMega());
document.addEventListener("click", (e)=> { if(!megaOpen) return; if(!startLearningPanel.contains(e.target) && !startLearningBtn.contains(e.target)) closeMega(); });
if (startLearningPanel) startLearningPanel.addEventListener("click", (e)=> {
  const btn = e.target.closest(".pill-btn");
  if (!btn) return;
  const action = btn.dataset.megaAction;
  if (action === "courses") { setActiveView("courses-view"); const sideMatch = Array.from(sideLinks).find((s) => s.dataset.view === "courses-view"); if (sideMatch) { sideLinks.forEach((s) => s.classList.remove("active")); sideMatch.classList.add("active"); } loadCourses(); }
  else if (action === "blogs") alert("Blogs placeholder");
  else if (action === "notes") alert("Notes placeholder");
});

// ------- Dashboard loading & render (unchanged except calls to schedule loader) -------
async function loadDashboard() {
  if (!currentUser) return;
  const resp = await api(`/api/dashboard/${currentUser.id}`);
  dashboardData = resp;
  if (currentUser.name && avatarInitial) avatarInitial.textContent = currentUser.name.charAt(0).toUpperCase();
  if (profileNameEl) profileNameEl.textContent = resp.user.name;
  if (profileEmailEl) profileEmailEl.textContent = currentUser.email;
  if (profileRoleEl) profileRoleEl.textContent = resp.user.role;
  if (profileInterestsEl) profileInterestsEl.textContent = resp.user.interests && resp.user.interests.length ? resp.user.interests.join(", ") : "Not set";
  // Overview
  renderCoursesOverview(resp.coursesOverview);
  // overall percent
  const overallPercent = computeOverallPercent(resp.coursesOverview);
  if (overallDonut) overallDonut.style.setProperty("--value", overallPercent);
  if (overallPercentEl) overallPercentEl.textContent = `${overallPercent}%`;
  if (weeklyHoursEl) weeklyHoursEl.textContent = `You studied ${resp.summary.weeklyHours} hours this week.`;
  if (statusTextEl) statusTextEl.textContent = resp.summary.status;
  if (heroWeekHoursEl) heroWeekHoursEl.textContent = `${resp.summary.weeklyHours}h`;
  if (heroCoursesProgressEl) heroCoursesProgressEl.textContent = resp.coursesOverview.filter((c) => c.progressPercent > 0 && c.progressPercent < 100).length;
  renderRecentBars(resp.coursesOverview);
  renderHeatmap(resp.summary.activity);
  renderTodos(resp.todos);
  renderRecommended(resp.recommended);
  renderAnnouncements(resp.announcements);

  // load schedule (today + week strip)
  await loadScheduleForCurrentUser();
}

// --- existing rendering functions (kept as-is) ---
function renderCoursesOverview(list) {
  if (!coursesOverviewEl) return;
  coursesOverviewEl.innerHTML = "";

  list.forEach((c) => {
    const item = document.createElement("div");
    item.className = "course-item";

    const header = document.createElement("div");
    header.className = "course-item-header";

    const title = document.createElement("div");
    title.className = "course-item-title";
    title.textContent = humanizeFilename(c.title);
    title.style.whiteSpace = "normal";
    title.style.wordBreak = "break-word";
    title.style.lineHeight = "1.3";

    const percent = document.createElement("div");
    percent.style.fontSize = "0.78rem";
    percent.style.color = "#9ca3af";
    percent.textContent = `${c.progressPercent}%`;

    header.appendChild(title);
    header.appendChild(percent);

    const desc = document.createElement("div");
    desc.className = "course-item-desc";
    desc.textContent = c.description || "";
    desc.style.whiteSpace = "normal";
    desc.style.wordBreak = "break-word";

    const actions = document.createElement("div");
    actions.className = "course-item-actions";

    const track = document.createElement("div");
    track.className = "progress-bar-track";

    const fill = document.createElement("div");
    fill.className = "progress-bar-fill";
    fill.style.width = `${c.progressPercent}%`;

    track.appendChild(fill);

    const btn = document.createElement("button");
    btn.className = "primary-btn tiny";
    btn.textContent = "Continue";
    btn.addEventListener("click", () => {
      setActiveView("courses-view");
      const sideMatch = Array.from(sideLinks).find(
        (s) => s.dataset.view === "courses-view"
      );
      if (sideMatch) {
        sideLinks.forEach((s) => s.classList.remove("active"));
        sideMatch.classList.add("active");
      }
      loadCourses(c.id);
    });

    actions.appendChild(track);
    actions.appendChild(btn);

    item.appendChild(header);
    item.appendChild(desc);
    item.appendChild(actions);

    coursesOverviewEl.appendChild(item);
  });
}


function computeOverallPercent(list) {
  if (!list || list.length === 0) return 0;
  const sum = list.reduce((acc, c) => acc + c.progressPercent, 0);
  return Math.round(sum / list.length);
}

function renderRecentBars(list) {
  if (!recentBarsEl) return;
  recentBarsEl.innerHTML = "";
  const sorted = [...list].sort((a, b) => new Date(b.lastViewedAt || 0) - new Date(a.lastViewedAt || 0));
  sorted.slice(0, 3).forEach((c) => {
    const row = document.createElement("div"); row.className = "recent-row";
    const label = document.createElement("span"); label.textContent = c.title; label.style.flex = "0 0 120px";
    const track = document.createElement("div"); track.className = "progress-bar-track"; const fill = document.createElement("div"); fill.className = "progress-bar-fill"; fill.style.width = `${c.progressPercent}%`; track.appendChild(fill);
    row.appendChild(label); row.appendChild(track); recentBarsEl.appendChild(row);
  });
}

// Heatmap implementation (kept)
function renderHeatmap(activityTimestamps) {
  if (!heatmapGridEl || !heatmapMonthLabelsEl) return;
  heatmapGridEl.innerHTML = "";
  heatmapMonthLabelsEl.innerHTML = "";
  const activity = activityTimestamps || [];
  const counts = {};
  activity.forEach((ts) => {
    const d = new Date(ts).toISOString().slice(0, 10);
    counts[d] = (counts[d] || 0) + 1;
  });
  const today = new Date();
  const startDate = new Date(today);
  startDate.setDate(today.getDate() - 7 * 53 + 1);
  const monthLabels = []; let lastMonth = null;
  for (let week = 0; week < 53; week++) {
    const firstDayOfWeek = new Date(startDate);
    firstDayOfWeek.setDate(startDate.getDate() + week * 7);
    const month = firstDayOfWeek.toLocaleString("default", { month: "short" });
    if (month !== lastMonth) { monthLabels.push(month); lastMonth = month; } else monthLabels.push("");
  }
  monthLabels.forEach((m) => { const span = document.createElement("span"); span.textContent = m; heatmapMonthLabelsEl.appendChild(span); });
  const daysPerWeek = 7;
  for (let dayIndex = 0; dayIndex < daysPerWeek * 53; dayIndex++) {
    const current = new Date(startDate);
    current.setDate(startDate.getDate() + dayIndex);
    const iso = current.toISOString().slice(0, 10);
    const count = counts[iso] || 0;
    const div = document.createElement("div"); div.className = "heat-square";
    let levelClass = "";
    if (count === 0) levelClass = ""; else if (count === 1) levelClass = "heat-level-1"; else if (count === 2) levelClass = "heat-level-2"; else if (count === 3) levelClass = "heat-level-3"; else levelClass = "heat-level-4";
    if (levelClass) div.classList.add(levelClass);
    div.title = count === 0 ? `${iso}: No activity` : `${iso}: Logged in (${count} time${count > 1 ? "s" : ""})`;
    div.addEventListener("click", () => {
      if (count === 0) alert(`${iso}: No activity recorded.`); else alert(`${iso}: You logged in ${count} time(s).`);
    });
    heatmapGridEl.appendChild(div);
  }
}

// Todos
function renderTodos(todos) {
  if (!todoListEl) return;
  todoListEl.innerHTML = "";
  (todos || []).forEach((text, idx) => {
    const li = document.createElement("li");
    const label = document.createElement("label");
    label.style.display = "flex"; label.style.alignItems = "center"; label.style.gap = "0.4rem";
    const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.dataset.index = idx;
    const span = document.createElement("span"); span.textContent = text;
    label.appendChild(checkbox); label.appendChild(span); li.appendChild(label); todoListEl.appendChild(li);
  });
}
if (saveTodosBtn) {
  saveTodosBtn.addEventListener("click", async () => {
    if (!currentUser || !dashboardData) return;
    const todos = [];
    todoListEl.querySelectorAll("li span").forEach((span) => todos.push(span.textContent));
    try {
      await api(`/api/todos/${currentUser.id}`, { method: "POST", body: JSON.stringify({ todos }) });
      alert("Plan saved (stored in memory on the server).");
    } catch (err) { alert("Failed to save plan: " + err.message); }
  });
}

function renderRecommended(list) { if (!recommendedEl) return; recommendedEl.innerHTML = ""; (list || []).forEach((c) => { const div = document.createElement("div"); div.className = "recommended-item"; div.textContent = `${c.title} · ${c.category} · ${c.level}`; recommendedEl.appendChild(div); }); }
function renderAnnouncements(list) { if (!announcementsEl) return; announcementsEl.innerHTML = ""; (list || []).forEach((a) => { const li = document.createElement("li"); li.textContent = `${a.message} (${a.date})`; announcementsEl.appendChild(li); }); }

// ------- Courses view -------
async function loadCourses(focusCourseId) {
  if (!allCoursesCache.length) {
    const resp = await api("/api/courses");
    allCoursesCache = resp;
  }
  renderCourseCards(allCoursesCache);
  if (focusCourseId) showCourseDetail(focusCourseId);
}

function renderCourseCards(list) {
  if (!coursesListEl) return;
  coursesListEl.innerHTML = "";
  list.forEach((c) => {
    const card = document.createElement("div"); card.className = "course-card";
    const title = document.createElement("div"); title.className = "course-card-title"; title.textContent = c.title;
    const desc = document.createElement("div"); desc.className = "course-card-desc"; desc.textContent = c.description;
    const tags = document.createElement("div"); tags.className = "course-tag-row";
    const catTag = document.createElement("span"); catTag.className = "tag"; catTag.textContent = c.category;
    const levelTag = document.createElement("span"); levelTag.className = "tag"; levelTag.textContent = c.level;
    const paidTag = document.createElement("span"); paidTag.className = "tag " + (c.isPaid ? "paid" : "free"); paidTag.textContent = c.isPaid ? "Paid" : "Free";
    tags.appendChild(catTag); tags.appendChild(levelTag); tags.appendChild(paidTag);
    const btn = document.createElement("button"); btn.className = "primary-btn tiny"; btn.textContent = "View details"; btn.addEventListener("click", () => showCourseDetail(c.id));
    card.appendChild(title); card.appendChild(desc); card.appendChild(tags); card.appendChild(btn); coursesListEl.appendChild(card);
  });
}

// showCourseDetail: robust encoding of filename (encode only basename)
async function showCourseDetail(courseId) {
  try {
    const course = await api(`/api/courses/${courseId}`);

    courseDetailTitle.textContent = humanizeFilename(course.title);
    courseDetailDesc.textContent = course.description;
    courseDetailMeta.textContent =
      `${course.category} · ${course.level} · Estimated ${course.estimatedHours} hours`;

    courseModulesEl.innerHTML = "";

    // ================= VIDEO PLAYER =================
    if (course.media_type === "video" && course.video_url) {
      const youTubeId = extractYouTubeId(course.video_url);

      const playerWrap = document.createElement("div");
      playerWrap.className = "course-video-wrap";

      // ---- status labels (shared by both player types) ----
      const status = document.createElement("div");
      status.className = "video-status";
      status.style.marginTop = "8px";
      status.style.fontSize = "0.9rem";
      status.style.color = "var(--muted)";
      status.textContent = "Loading video...";

      const focusStatus = document.createElement("div");
      focusStatus.className = "video-focus-status";
      focusStatus.style.marginTop = "6px";
      focusStatus.style.fontSize = "0.9rem";
      focusStatus.textContent = "Focus status: camera off";

      // ---- CAMERA TOGGLE BUTTON (shared) ----
      // `isCurrentlyPlaying()` is filled in below by whichever player type
      // is used, so this button doesn't need to know which one it is.
      let isCurrentlyPlaying = () => false;
      const cameraToggleBtn = document.createElement("button");
      cameraToggleBtn.className = "secondary-btn tiny";
      cameraToggleBtn.style.marginTop = "8px";
      cameraToggleBtn.textContent = "Camera: OFF";

      cameraToggleBtn.onclick = () => {
        cameraEnabled = !cameraEnabled;

        if (!cameraEnabled) {
          cameraToggleBtn.textContent = "Camera: OFF";
          focusStatus.textContent = "Focus status: camera off";
          stopFocusDetection();
        } else {
          cameraToggleBtn.textContent = "Camera: ON";
          if (isCurrentlyPlaying()) {
            startFocusDetection(focusStatus);
          }
        }
      };

      if (youTubeId) {
        // ---- YOUTUBE PLAYER (unlisted upload) ----
        const playerHost = document.createElement("div");
        playerHost.id = `yt-player-${course.id}-${Date.now()}`;
        playerHost.style.width = "100%";
        playerHost.style.aspectRatio = "16 / 9";
        playerHost.style.borderRadius = "8px";
        playerWrap.appendChild(playerHost);

        // Proxy object so other code (e.g. the nav camera-toggle button)
        // that reads `currentVideoEl.paused` keeps working unmodified,
        // without needing to know this is a YouTube embed under the hood.
        let ytPlayer = null;
        isCurrentlyPlaying = () => !!ytPlayer && ytPlayer.getPlayerState() === YT.PlayerState.PLAYING;
        const videoProxy = { get paused() { return !isCurrentlyPlaying(); } };

        loadYouTubeApi().then((YT) => {
          ytPlayer = new YT.Player(playerHost.id, {
            videoId: youTubeId,
            playerVars: { playsinline: 1 },
            events: {
              onReady: (e) => {
                const dur = e.target.getDuration();
                status.textContent = dur ? `Duration: ${Math.round(dur)}s` : "Ready";
              },
              onStateChange: (e) => {
                if (e.data === YT.PlayerState.PLAYING) {
                  currentVideoEl = videoProxy;
                  currentFocusStatusEl = focusStatus;
                  if (cameraEnabled) {
                    focusStatus.textContent = "Starting focus detection…";
                    startFocusDetection(focusStatus);
                  }
                } else if (e.data === YT.PlayerState.PAUSED) {
                  stopFocusDetection();
                  if (cameraEnabled) focusStatus.textContent = "Focus detection paused";
                } else if (e.data === YT.PlayerState.ENDED) {
                  stopFocusDetection();
                  focusStatus.textContent = "Video ended";
                }
              },
            },
          });
        });
      } else {
        // ---- LOCAL FILE PLAYER (served from this app's own /static/videos) ----
        const urlParts = course.video_url.split("/");
        const basename = urlParts[urlParts.length - 1] || "";
        const encodedBasename = encodeURIComponent(decodeURIComponent(basename));
        const safeSrc = `/static/videos/${encodedBasename}`;

        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.preload = "metadata";
        videoEl.playsInline = true;
        videoEl.style.width = "100%";
        videoEl.style.borderRadius = "8px";
        videoEl.src = safeSrc;

        isCurrentlyPlaying = () => !videoEl.paused;

        videoEl.addEventListener("loadedmetadata", () => {
          const dur = isFinite(videoEl.duration)
            ? `${Math.round(videoEl.duration)}s`
            : "unknown";
          status.textContent = `Duration: ${dur}`;
        });

        videoEl.addEventListener("playing", () => {
          currentVideoEl = videoEl;
          currentFocusStatusEl = focusStatus;

          if (cameraEnabled) {
            focusStatus.textContent = "Starting focus detection…";
            startFocusDetection(focusStatus);
          }
        });

        videoEl.addEventListener("pause", () => {
          stopFocusDetection();
          if (cameraEnabled) {
            focusStatus.textContent = "Focus detection paused";
          }
        });

        videoEl.addEventListener("ended", () => {
          stopFocusDetection();
          focusStatus.textContent = "Video ended";
        });

        videoEl.addEventListener("error", (e) => {
          console.error("Video load error:", e);
          status.textContent = "Unable to load video.";
        });

        playerWrap.appendChild(videoEl);
      }

      // ---- APPEND (shared) ----
      playerWrap.appendChild(status);
      playerWrap.appendChild(focusStatus);
      playerWrap.appendChild(cameraToggleBtn);
      courseModulesEl.appendChild(playerWrap);
    }

    // ================= MODULE LIST =================
    const ul = document.createElement("ul");
    ul.style.listStyle = "none";
    ul.style.paddingLeft = "0";
    ul.style.marginTop = "1rem";

    (course.modules || []).forEach((m) => {
      const li = document.createElement("li");
      li.style.display = "flex";
      li.style.justifyContent = "space-between";
      li.style.alignItems = "center";
      li.style.marginBottom = "0.5rem";

      const left = document.createElement("span");
      left.textContent = `${m.order}. ${m.title}`;

      const btn = document.createElement("button");
      btn.className = "secondary-btn tiny";
      btn.textContent = "Mark complete";
      btn.onclick = () => markModuleComplete(course.id, m.id);

      li.appendChild(left);
      li.appendChild(btn);
      ul.appendChild(li);
    });

    courseModulesEl.appendChild(ul);
    courseModulesEl.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    console.error(err);
    courseDetailDesc.textContent = err.message || "Failed to load course.";
  }
}

async function markModuleComplete(courseId, moduleId) {
  if (!currentUser) return;
  try {
    await api(`/api/courses/${courseId}/complete`, {
      method: "POST",
      body: JSON.stringify({ userId: currentUser.id, moduleId }),
    });
    await loadDashboard();
    alert("Module marked as complete.");
  } catch (err) {
    alert("Unable to update progress: " + err.message);
  }
}

// ------- Hero search + auto-login (unchanged) -------
if (heroSearchBtn) {
  heroSearchBtn.addEventListener("click", () => {
    const query = document.getElementById("hero-search-input").value.toLowerCase();
    const domain = heroDomainSelect.value;
    setActiveView("courses-view");
    const sideMatch = Array.from(sideLinks).find((s) => s.dataset.view === "courses-view");
    if (sideMatch) {
      sideLinks.forEach((s) => s.classList.remove("active"));
      sideMatch.classList.add("active");
    }
    if (allCoursesCache.length) {
      const filtered = allCoursesCache.filter((c) => {
        const matchesDomain = !domain || c.category === domain;
        const matchesQuery = !query || c.title.toLowerCase().includes(query) || c.description.toLowerCase().includes(query);
        return matchesDomain && matchesQuery;
      });
      renderCourseCards(filtered);
    } else {
      loadCourses();
    }
  });
}

// ------------------ SCHEDULE / TASKS FRONTEND ------------------
// Helper: convert ISO UTC to local date & time strings
function isoToLocalParts(iso) {
  if (!iso) return { date: "", time: "" };
  try {
    const d = new Date(iso);
    if (isNaN(d)) return { date: "", time: "" };
    const date = d.toLocaleDateString();
    const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return { date, time, rawDateObj: d };
  } catch (e) { return { date: "", time: "" }; }
}

function toInputDateValues(iso) {
  if (!iso) {
    // default to today +1h
    const defaultDt = new Date();
    defaultDt.setHours(defaultDt.getHours() + 1);
    const yyyy = defaultDt.toISOString().slice(0,10);
    const hhmm = defaultDt.toTimeString().slice(0,5);
    return { date: yyyy, time: hhmm };
  }
  const d = new Date(iso);
  // should produce local date yyyy-mm-dd and local hh:mm
  const localISO = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString();
  return { date: localISO.slice(0,10), time: localISO.slice(11,16) };
}

function localDateTimeToUTC_ISO(dateStr, timeStr) {
  // dateStr: yyyy-mm-dd, timeStr: HH:MM (24h) -> returns ISO UTC string with Z
  if (!dateStr) return null;
  const dtLocal = new Date(`${dateStr}T${timeStr || "00:00"}:00`);
  // convert to UTC ISO
  const isoUtc = new Date(dtLocal.getTime() - dtLocal.getTimezoneOffset() * 60000).toISOString();
  return isoUtc.endsWith("Z") ? isoUtc : isoUtc + "Z";
}

// Load schedule for current user (default next 7 days)
async function loadScheduleForCurrentUser(days = 7) {
  if (!currentUser) return;
  try {
    const resp = await api(`/api/schedule/${currentUser.id}?days=${days}`);
    const tasks = resp.tasks || [];
    scheduleCache = tasks; // cache
    renderWeekStrip(tasks);
    renderTodayTasks(tasks);
    scheduleClientNotifications(tasks);
  } catch (err) {
    console.error("Unable to load schedule:", err.message);
  }
}

// Render 7-day strip (shows counts and clickable days)
function renderWeekStrip(tasks) {
  if (!weekStripEl) return;
  weekStripEl.innerHTML = "";
  const today = new Date();
  // create 7-day array starting today
  for (let i = 0; i < 7; i++) {
    const d = new Date(today);
    d.setDate(today.getDate() + i);
    const isoDay = d.toISOString().slice(0,10);
    const dayTasks = tasks.filter(t => {
      try { return new Date(t.dueAt).toISOString().slice(0,10) === isoDay; } catch { return false; }
    });
    const completed = dayTasks.filter(t => t.status === "done").length;
    const card = document.createElement("button");
    card.className = "day-card";
    card.type = "button";
    card.innerHTML = `<div class="day-label">${d.toLocaleDateString(undefined, { weekday: 'short' })}</div>
                      <div class="day-num">${d.getDate()}</div>
                      <div class="day-count">${dayTasks.length} tasks</div>`;
    if (completed > 0) {
      const small = document.createElement("div");
      small.className = "day-complete";
      small.textContent = `${completed} done`;
      card.appendChild(small);
    }
    card.addEventListener("click", () => {
      // filter today's tasks in UI
      renderTodayTasks(tasks, isoDay);
    });
    weekStripEl.appendChild(card);
  }
}

// Render Today's Plan card (optionally for a specific date isoDay 'YYYY-MM-DD')
function renderTodayTasks(tasks, isoDay = null) {
  if (!todayTasksEl) return;
  todayTasksEl.innerHTML = "";
  const today = isoDay || new Date().toISOString().slice(0,10);
  const todays = tasks.filter(t => {
    try { return new Date(t.dueAt).toISOString().slice(0,10) === today; } catch { return false; }
  }).sort((a,b) => new Date(a.dueAt) - new Date(b.dueAt));

  if (todays.length === 0) {
    const empty = document.createElement("div"); empty.className = "empty-today"; empty.textContent = "No tasks scheduled for today — add one to keep your study streak!";
    todayTasksEl.appendChild(empty);
    return;
  }

  todays.forEach((t) => {
    const li = document.createElement("div");
    li.className = "task-row";

    const left = document.createElement("div");
    left.className = "task-left";
    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = (t.status === "done");
    chk.addEventListener("change", async () => {
      if (!currentUser) return;
      if (chk.checked) {
        try {
          await api(`/api/schedule/${currentUser.id}/${t.id}/complete`, { method: "POST" });
          await loadScheduleForCurrentUser();
        } catch (err) { alert("Failed to mark complete: " + err.message); chk.checked = false; }
      } else {
        // un-complete -> update status to todo via PUT
        try {
          await api(`/api/schedule/${currentUser.id}/${t.id}`, { method: "PUT", body: JSON.stringify({ status: "todo" }) });
          await loadScheduleForCurrentUser();
        } catch (err) { alert("Failed to update task: " + err.message); chk.checked = true; }
      }
    });

    const metaSpan = document.createElement("div");
    metaSpan.className = "task-meta";
    const parts = isoToLocalParts(t.dueAt);
    metaSpan.innerHTML = `<div class="task-title">${t.title}</div><div class="task-sub small">${parts.time} · ${t.durationMinutes || 0}m · ${t.priority || 'medium'}</div>`;

    left.appendChild(chk);
    left.appendChild(metaSpan);

    const right = document.createElement("div");
    right.className = "task-actions";
    const editBtn = document.createElement("button"); editBtn.className = "icon-btn"; editBtn.title = "Edit"; editBtn.textContent = "✎";
    const snoozeBtn = document.createElement("button"); snoozeBtn.className = "icon-btn"; snoozeBtn.title = "Snooze"; snoozeBtn.textContent = "🕓";
    const delBtn = document.createElement("button"); delBtn.className = "icon-btn"; delBtn.title = "Delete"; delBtn.textContent = "🗑";

    editBtn.addEventListener("click", () => openTaskModal(t));
    snoozeBtn.addEventListener("click", () => showSnoozeOptions(t));
    delBtn.addEventListener("click", async () => {
      if (!currentUser) return;
      if (!confirm("Delete this task?")) return;
      try {
        await api(`/api/schedule/${currentUser.id}/${t.id}`, { method: "DELETE" });
        await loadScheduleForCurrentUser();
      } catch (err) { alert("Failed to delete: " + err.message); }
    });

    right.appendChild(editBtn); right.appendChild(snoozeBtn); right.appendChild(delBtn);

    li.appendChild(left); li.appendChild(right);
    todayTasksEl.appendChild(li);
  });
}

// Open task modal for create or edit
function openTaskModal(task = null) {
  if (!taskModal) return;
  taskModal.classList.remove("hidden");
  if (taskModalTitle) taskModalTitle.textContent = task ? "Edit Task" : "Add Task";
  if (taskFormTitle) taskFormTitle.value = task ? task.title : "";
  if (taskFormNotes) taskFormNotes.value = task ? task.notes : "";
  const { date, time } = toInputDateValues(task ? task.dueAt : null);
  if (taskFormDate) taskFormDate.value = date;
  if (taskFormTime) taskFormTime.value = time;
  if (taskFormDuration) taskFormDuration.value = task ? (task.durationMinutes || 0) : 30;
  if (taskFormPriority) taskFormPriority.value = task ? (task.priority || "medium") : "medium";
  if (taskFormRecurrence) taskFormRecurrence.value = task && task.recurrence ? `${task.recurrence.freq}|${task.recurrence.interval}` : "";
  taskModal.dataset.editTaskId = task ? task.id : "";
  if (taskModalMsg) taskModalMsg.textContent = "";
  // focus
  taskFormTitle && taskFormTitle.focus();
}

// Close modal
function closeTaskModal() {
  if (!taskModal) return;
  taskModal.classList.add("hidden");
  taskModal.dataset.editTaskId = "";
  if (taskForm) taskForm.reset();
}

// Save (create or update)
if (taskFormSave) {
  taskFormSave.addEventListener("click", async (e) => {
    e.preventDefault();
    if (!currentUser) { alert("Please login"); return; }
    const title = taskFormTitle.value.trim();
    if (!title) { if (taskModalMsg) taskModalMsg.textContent = "Title is required."; return; }
    const notes = taskFormNotes.value;
    const date = taskFormDate.value;
    const time = taskFormTime.value;
    const dueAt = localDateTimeToUTC_ISO(date, time);
    const durationMinutes = parseInt(taskFormDuration.value || "0", 10);
    const priority = taskFormPriority.value || "medium";
    const recVal = taskFormRecurrence.value;
    let recurrence = null;
    if (recVal) {
      const parts = recVal.split("|");
      recurrence = { freq: parts[0], interval: parseInt(parts[1] || "1", 10) || 1 };
    }
    const payload = { title, notes, dueAt, durationMinutes, priority, recurrence };
    const editId = taskModal.dataset.editTaskId;
    try {
      if (editId) {
        await api(`/api/schedule/${currentUser.id}/${editId}`, { method: "PUT", body: JSON.stringify(payload) });
      } else {
        await api(`/api/schedule/${currentUser.id}`, { method: "POST", body: JSON.stringify(payload) });
      }
      closeTaskModal();
      await loadScheduleForCurrentUser();
    } catch (err) {
      if (taskModalMsg) taskModalMsg.textContent = err.message || "Failed to save task.";
    }
  });
}
if (taskFormCancel) taskFormCancel.addEventListener("click", (e) => { e.preventDefault(); closeTaskModal(); });

// Add task quick button
if (addTaskBtn) addTaskBtn.addEventListener("click", () => openTaskModal(null));

// Snooze quick options UI
function showSnoozeOptions(task) {
  const minutes = prompt("Snooze by minutes (e.g. 10, 30, 60):", "30");
  if (minutes === null) return;
  const m = parseInt(minutes, 10);
  if (!Number.isFinite(m) || m <= 0) { alert("Invalid minutes"); return; }
  snoozeTask(task.id, m);
}

async function snoozeTask(taskId, minutes) {
  if (!currentUser) return;
  try {
    await api(`/api/schedule/${currentUser.id}/${taskId}/snooze`, { method: "POST", body: JSON.stringify({ minutes }) });
    await loadScheduleForCurrentUser();
  } catch (err) {
    alert("Failed to snooze: " + err.message);
  }
}

// Notifications: schedule client-side notifications for tasks due soon
function clearScheduledNotifications() {
  notificationTimers.forEach(id => clearTimeout(id));
  notificationTimers = [];
}

function scheduleClientNotifications(tasks) {
  clearScheduledNotifications();
  if (!("Notification" in window)) return;
  // choose tasks due within next 24 hours and that are not done
  const now = Date.now();
  tasks.forEach(t => {
    try {
      if (t.status === "done") return;
      const due = new Date(t.dueAt).getTime();
      // schedule a notification 10 minutes before due (if in future)
      const notifyBeforeMs = 10 * 60 * 1000;
      const notifyAt = due - notifyBeforeMs;
      if (notifyAt > now) {
        const timeout = setTimeout(() => {
          showInAppToast(`Reminder: "${t.title}" is due in 10 minutes.`);
          if (Notification.permission === "granted") {
            new Notification("Study reminder", { body: `${t.title} — due at ${new Date(t.dueAt).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}` });
          }
        }, notifyAt - now);
        notificationTimers.push(timeout);
      }
    } catch (e) { /* ignore malformed dates */ }
  });
}

// helper: show minimal in-app toast
function showInAppToast(msg) {
  const toast = document.createElement("div");
  toast.className = "inapp-toast";
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.classList.add("visible"), 50);
  setTimeout(() => { toast.classList.remove("visible"); setTimeout(()=>toast.remove(),300); }, 5500);
}

// Request Notification permission at first interaction (deferred)
async function ensureNotificationPermission() {
  if (!("Notification" in window)) return;
  if (Notification.permission === "default") {
    try { await Notification.requestPermission(); } catch (e) { /* ignore */ }
  }
}

// Hook: ask for permission when schedule view opens or when the user logs in
if (scheduleViewEl) {
  scheduleViewEl.addEventListener("mouseenter", () => ensureNotificationPermission());
}

// When user logs in from storage on load, ensure notifications scheduled
window.addEventListener("DOMContentLoaded", async () => {
  const saved = localStorage.getItem("lmsUser");
  if (saved) {
    try {
      currentUser = JSON.parse(saved);
      await loadDashboard();
      authContainer && authContainer.classList.add("hidden");
      appContainer && appContainer.classList.remove("hidden");
      ensureNotificationPermission();
    } catch (e) {
      localStorage.removeItem("lmsUser");
    }
  }
});

// expose manual function to refresh schedule (useful in console)
window.refreshSchedule = () => loadScheduleForCurrentUser();

// ================= Focus Detection Helpers =================
async function startFocusDetection(statusEl) {
  console.log("▶️ startFocusDetection");

  currentFocusStatusEl = statusEl;
  if (!cameraEnabled) return;

  // HARD RESET (prevents ghost camera)
  stopFocusDetection();

  try {
    focusStream = await navigator.mediaDevices.getUserMedia({ video: true });

    cameraPreviewEl = document.createElement("div");
    cameraPreviewEl.className = "camera-preview";

    focusCameraVideo = document.createElement("video");
    focusCameraVideo.autoplay = true;
    focusCameraVideo.muted = true;
    focusCameraVideo.playsInline = true;
    focusCameraVideo.srcObject = focusStream;

    cameraPreviewEl.appendChild(focusCameraVideo);
    document.body.appendChild(cameraPreviewEl);

    await focusCameraVideo.play();

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    focusInterval = setInterval(async () => {
      if (!cameraEnabled || !focusCameraVideo?.videoWidth) return;

      canvas.width = focusCameraVideo.videoWidth;
      canvas.height = focusCameraVideo.videoHeight;
      ctx.drawImage(focusCameraVideo, 0, 0);

      const image = canvas.toDataURL("image/jpeg");

      try {
        const res = await api("/api/focus/analyze", {
          method: "POST",
          body: JSON.stringify({ image }),
        });

        // Only report whether the person is actually looking at the screen
        // (face present, eyes open) -- deliberately ignore the emotion
        // sub-state (frustrated/confused/bored/neutral) here. Being
        // frustrated or confused doesn't mean someone isn't focusing; it
        // would be misleading to show anything other than "Focused" as
        // long as they're actually present and attentive.
        if (res.focused === false) {
          statusEl.textContent = `⚠️ Not focused: ${res.reason}`;
          statusEl.style.color = "red";
        } else {
          statusEl.textContent = "✅ Focused";
          statusEl.style.color = "green";
        }
      } catch (err) {
        console.error("focus/analyze failed:", err);
        statusEl.textContent = `⚠️ Focus check failed: ${err.message}`;
        statusEl.style.color = "orange";
      }
    }, 2000);

  } catch (err) {
    console.error(err);
    statusEl.textContent = "Camera permission denied";
  }
}

function stopFocusDetection() {
  console.log("🛑 stopFocusDetection");

  // Stop focus interval
  if (focusInterval) {
    clearInterval(focusInterval);
    focusInterval = null;
  }

  // Stop camera stream
  if (focusStream) {
    focusStream.getTracks().forEach(track => track.stop());
    focusStream = null;
  }

  // Stop and clean video element
  if (focusCameraVideo) {
    focusCameraVideo.pause();
    focusCameraVideo.srcObject = null;
    focusCameraVideo = null;
  }

  // Remove camera preview UI
  if (cameraPreviewEl) {
    cameraPreviewEl.remove();
    cameraPreviewEl = null;
  }

  // Reset focus state
  focusBlocked = false;

  // Hide popup if visible
  if (typeof hideNotFocusedPopup === "function") {
    hideNotFocusedPopup();
  }

  console.log("✅ Focus detection fully stopped and reset");
}

function createCameraToggle() {
  const btn = document.createElement("button");
  btn.textContent = "Camera: OFF";
  btn.className = "secondary-btn tiny";
  btn.style.marginTop = "8px";

  btn.onclick = () => {
    cameraEnabled = !cameraEnabled;

    if (!cameraEnabled) {
      btn.textContent = "Camera: OFF";
      stopFocusDetection();
    } else {
      btn.textContent = "Camera: ON";
      if (currentVideoEl && !currentVideoEl.paused) {
        startFocusDetection(currentFocusStatusEl);
      }
    }
  };

  return btn;
}

// ================= Focus Popup Helpers =================
function showNotFocusedPopup() {
  let popup = document.getElementById("focus-popup");

  if (!popup) {
    popup = document.createElement("div");
    popup.id = "focus-popup";
    popup.textContent = "⚠️ You are not focused. Video paused.";
    document.body.appendChild(popup);
  }

  popup.classList.add("show");
}

function hideNotFocusedPopup() {
  const popup = document.getElementById("focus-popup");
  if (popup) popup.classList.remove("show");
}
// =======================================================
