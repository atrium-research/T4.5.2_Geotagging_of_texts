(() => {
  const $ = (selector) => document.querySelector(selector);
  const fileInput = $('#fileInput');
  const form = $('#uploadForm');
  const dropzone = $('#dropzone');
  const runButton = $('#runButton');
  const progressSection = $('#progressSection');
  const errorBox = $('#errorBox');
  const resultsSection = $('#resultsSection');
  const stepNames = ['extract', 'ner', 'context', 'retrieve', 'rerank', 'tei'];
  const stageMap = {queued: 0, extract: 0, gazetteer: 0, ner: 1, context: 2, retrieve: 3, rerank: 4, coordinates: 5, descriptions: 5, tei: 5, complete: 6, failed: 6};
  let currentJob = null;
  let currentResults = null;
  let selectedMentionId = null;
  let selectedLocationId = null;
  let map = null;
  let markersByLocation = new Map();
  let mapLayer = null;
  let pollTimer = null;
  let currentEditMode = null;
  let pendingSelection = null;
  // Paging keeps long documents readable. shortcut: a single paragraph longer than TEXT_PAGE_CHARS gets its own page, uncut.
  const PAGE_SIZE = 50, TEXT_PAGE_CHARS = 6000;
  let textPages = [[0, 0]];   // [firstParagraph, endParagraph) per text page
  let textPage = 0, tablePage = 0;

  // Theme is intentionally client-side only; no account or preference service is needed.
  const savedTheme = (() => { try { return localStorage.getItem('fieldnotes-theme'); } catch (_) { return null; } })();
  document.documentElement.dataset.theme = savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  $('#themeToggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('fieldnotes-theme', next); } catch (_) {}
    refreshMapTiles();
  });

  $('#addModeButton').addEventListener('click', () => activateEditMode('add'));
  $('#modifyModeButton').addEventListener('click', () => activateEditMode('modify'));
  $('#deleteModeButton').addEventListener('click', () => activateEditMode('delete'));
  window.addEventListener('mouseup', captureSourceSelection);
  window.addEventListener('keyup', captureSourceSelection);

  function bytesLabel(size) {
    if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }
  function selectFile(file) {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt', 'xml', 'tei'].includes(ext)) {
      showError('Choose a PDF, TXT, or XML/TEI file.');
      fileInput.value = '';
      runButton.disabled = true;
      return;
    }
    $('#dropTitle').textContent = file.name;
    $('#dropSubtitle').textContent = `${bytesLabel(file.size)} · Ready to process`;
    dropzone.classList.add('has-file');
    runButton.disabled = false;
    errorBox.classList.add('hidden');
  }
  fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); }));
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    try { const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files; } catch (_) {}
    selectFile(file);
  });
  $('#retryButton').addEventListener('click', () => {
    errorBox.classList.add('hidden');
    fileInput.value = '';
    $('#dropTitle').textContent = 'Drop your source file here';
    $('#dropSubtitle').innerHTML = 'or <span>browse files</span> on your device';
    runButton.disabled = true;
    dropzone.classList.remove('has-file');
    fileInput.click();
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;
    resetForRun();
    const formData = new FormData();
    formData.append('file', file);
    runButton.disabled = true;
    runButton.querySelector('span').textContent = 'Starting…';
    try {
      const response = await fetch('/api/jobs', {method: 'POST', body: formData});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not start the workflow.');
      currentJob = payload.job_id;
      pollTimer = setInterval(pollJob, 1200);
      pollJob();
    } catch (error) {
      showError(error.message || 'Could not start the workflow.');
      runButton.disabled = false;
      runButton.querySelector('span').textContent = 'Start workflow';
    }
  });

  function resetForRun() {
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorBox.classList.add('hidden');
    $('#progressTitle').textContent = 'Preparing your document';
    $('#progressMessage').textContent = 'Submitting source file to the local workflow…';
    $('#progressPercent').innerHTML = '0<span>%</span>';
    $('#progressFill').style.width = '0%';
    $('#eventList').replaceChildren();
    $('#eventCount').textContent = '0 updates';
    document.querySelectorAll('.workflow-step').forEach((step) => step.classList.remove('active', 'done'));
    currentResults = null; selectedMentionId = null; selectedLocationId = null;
    currentEditMode = null; pendingSelection = null;
    setEditModeButtons();
    scrollToElement(progressSection, 20);
  }

  async function pollJob() {
    if (!currentJob) return;
    try {
      const response = await fetch(`/api/jobs/${currentJob}`, {cache: 'no-store'});
      const job = await response.json();
      if (!response.ok) throw new Error(job.detail || 'Could not read the workflow status.');
      renderProgress(job);
      if (job.status === 'completed') {
        clearInterval(pollTimer); pollTimer = null;
        textPage = 0; tablePage = 0;
        renderResults(job.result);
        runButton.disabled = false;
        runButton.querySelector('span').textContent = 'Run another document';
      } else if (job.status === 'failed') {
        clearInterval(pollTimer); pollTimer = null;
        showError(job.error || job.message || 'The workflow failed.');
        runButton.disabled = false;
        runButton.querySelector('span').textContent = 'Try again';
      }
    } catch (error) {
      $('#progressMessage').textContent = error.message || 'Connection issue while checking the workflow.';
    }
  }

  function renderProgress(job) {
    const value = Number(job.progress || 0);
    $('#progressPercent').innerHTML = `${value}<span>%</span>`;
    $('#progressFill').style.width = `${value}%`;
    $('#progressTitle').textContent = job.status === 'completed' ? 'Workflow complete' : job.status === 'failed' ? 'Workflow stopped' : 'Processing your document';
    $('#progressMessage').textContent = job.message || 'Working…';
    $('#progressSpinner').classList.toggle('stopped', ['completed', 'failed'].includes(job.status));
    let activeIndex = stageMap[job.stage] ?? 0;
    document.querySelectorAll('.workflow-step').forEach((step, index) => {
      step.classList.toggle('done', index < activeIndex || job.status === 'completed');
      step.classList.toggle('active', index === activeIndex && !['completed', 'failed'].includes(job.status));
    });
    const events = job.events || [];
    $('#eventCount').textContent = `${events.length} update${events.length === 1 ? '' : 's'}`;
    const eventList = $('#eventList');
    eventList.replaceChildren();
    events.slice(-18).reverse().forEach((event) => {
      const row = document.createElement('div'); row.className = 'event-row';
      const percent = document.createElement('span'); percent.textContent = `${event.percent}%`;
      const stage = document.createElement('span'); stage.textContent = event.stage;
      const message = document.createElement('span'); message.textContent = event.message;
      row.append(percent, stage, message); eventList.append(row);
    });
  }

  function showError(message) {
    errorBox.classList.remove('hidden');
    $('#errorText').textContent = message;
    progressSection.classList.remove('hidden');
    $('#progressSpinner').classList.add('stopped');
  }

  function renderResults(result) {
    currentResults = result;
    selectedMentionId = null;
    selectedLocationId = null;
    currentEditMode = null;
    pendingSelection = null;
    setEditModeButtons();
    resultsSection.classList.remove('hidden');
    $('#resultsSubtitle').textContent = `${result.filename} · Review the linked place entities before using the TEI export.`;
    document.querySelectorAll('.warning-banner').forEach((el) => el.remove());
    $('#sourceTitle').textContent = result.title || result.filename;
    $('#teiDownload').href = `/api/jobs/${result.job_id}/download`;
    $('#jsonDownload').href = `/api/jobs/${result.job_id}/results.json`;
    renderStats();
    textPages = buildTextPages();
    textPage = Math.min(textPage, textPages.length - 1);
    tablePage = Math.min(tablePage, Math.max(0, Math.ceil((result.mentions || []).length / PAGE_SIZE) - 1));
    renderSource();
    renderTable();
    initMap();
    renderLocationDetail(null, null);
    if (result.warnings && result.warnings.length) {
      const warning = document.createElement('div'); warning.className = 'warning-banner';
      warning.textContent = result.warnings.join(' ');
      $('#resultsSubtitle').parentElement.appendChild(warning);
    }
    scrollToElement(resultsSection, 22);
  }

  function renderStats() {
    if (!currentResults) return;
    const stats = currentResults.stats || {};
    const mentions = currentResults.mentions || [];
    const locations = currentResults.locations || [];
    const mapped = locations.filter((location) => hasCoordinates(location)).length;
    const disambiguated = mentions.filter((mention) => Boolean(mention.location?.url)).length;
    const computedStats = {
      paragraphs: (currentResults.paragraphs || []).length,
      mentions: mentions.length,
      disambiguated,
      mapped,
      unresolved: mentions.filter((mention) => mention.location?.type === 'Unresolved').length,
    };
    currentResults.stats = {...stats, ...computedStats};
    const statData = [
      ['Paragraphs', computedStats.paragraphs, '¶'],
      ['Place mentions', computedStats.mentions, '⌕'],
      ['Disambiguated', computedStats.disambiguated, '↗'],
      ['Mapped places', computedStats.mapped, '⌖'],
    ];
    const statGrid = $('#statGrid'); statGrid.replaceChildren();
    statData.forEach(([label, value, symbol]) => {
      const card = document.createElement('div'); card.className = 'stat-card';
      const icon = document.createElement('div'); icon.className = 'stat-icon'; icon.textContent = symbol;
      const wrap = document.createElement('div');
      const metric = document.createElement('div'); metric.className = 'stat-value'; metric.textContent = value;
      const caption = document.createElement('div'); caption.className = 'stat-label'; caption.textContent = label;
      wrap.append(metric, caption); card.append(icon, wrap); statGrid.append(card);
    });
    $('#readerMentionCount').textContent = `${computedStats.mentions} place mention${computedStats.mentions === 1 ? '' : 's'}`;
    $('#mappedCountLabel').textContent = `${computedStats.mapped} mapped`;
    $('#tableCount').textContent = `${computedStats.mentions} record${computedStats.mentions === 1 ? '' : 's'}`;
  }

  function buildTextPages() {
    const paragraphs = currentResults.paragraphs || [];
    const pages = []; let start = 0, size = 0;
    paragraphs.forEach((paragraph, i) => {
      const length = String(paragraph.text || '').length;
      if (size && size + length > TEXT_PAGE_CHARS) { pages.push([start, i]); start = i; size = 0; }
      size += length;
    });
    pages.push([start, paragraphs.length]);
    return pages;
  }
  function pageOfParagraph(index) { return Math.max(0, textPages.findIndex(([from, to]) => index >= from && index < to)); }
  function pageMentions() {
    const [from, to] = textPages[textPage] || [0, 0];
    return (currentResults?.mentions || []).filter((m) => m.paragraph_index >= from && m.paragraph_index < to);
  }
  function renderPager(element, page, total, go) {
    element.replaceChildren();
    element.classList.toggle('hidden', total <= 1);
    const arrow = (label, target, name) => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'small-icon-button';
      button.textContent = label; button.title = name; button.setAttribute('aria-label', name);
      button.disabled = target < 0 || target >= total; button.addEventListener('click', () => go(target));
      return button;
    };
    const info = document.createElement('span'); info.textContent = `Page ${page + 1} of ${total}`;
    element.append(arrow('‹', page - 1, 'Previous page'), info, arrow('›', page + 1, 'Next page'));
  }
  function setTextPage(page) {
    textPage = page;
    renderSource(); initMap();
    $('#sourceText').scrollTop = 0;
  }

  function renderSource() {
    const container = $('#sourceText'); container.replaceChildren();
    if (!currentResults) return;
    renderPager($('#sourcePager'), textPage, textPages.length, setTextPage);
    const grouped = new Map();
    (currentResults.mentions || []).forEach((mention) => {
      if (!grouped.has(mention.paragraph_index)) grouped.set(mention.paragraph_index, []);
      grouped.get(mention.paragraph_index).push(mention);
    });
    const [pageFrom, pageTo] = textPages[textPage] || [0, 0];
    (currentResults.paragraphs || []).slice(pageFrom, pageTo).forEach((paragraph, offset) => {
      const index = pageFrom + offset;
      const section = document.createElement('section'); section.className = 'source-paragraph'; section.id = `paragraph-${index}`;
      const number = document.createElement('span'); number.className = 'paragraph-number'; number.textContent = `PARAGRAPH ${String(index + 1).padStart(3, '0')}`;
      section.append(number);
      const text = String(paragraph.text || '');
      const spans = (grouped.get(paragraph.index) || []).slice().sort((a, b) => a.start - b.start || b.end - a.end);
      let cursor = 0;
      spans.forEach((mention) => {
        if (mention.start < cursor || mention.end <= mention.start || mention.end > text.length) return;
        section.append(document.createTextNode(text.slice(cursor, mention.start)));
        const token = document.createElement('button'); token.type = 'button'; token.className = 'place-token'; token.dataset.mentionId = mention.id;
        token.textContent = text.slice(mention.start, mention.end);
        if (mention.id === selectedMentionId) token.classList.add('active');
        token.title = `${mention.location?.name || 'Unresolved place'} · click to inspect`;
        token.addEventListener('click', () => selectMention(mention.id, true));
        section.append(token);
        cursor = mention.end;
      });
      section.append(document.createTextNode(text.slice(cursor)));
      container.append(section);
    });
  }

  function renderTable() {
    const tbody = $('#mentionRows'); tbody.replaceChildren();
    const mentions = currentResults?.mentions || [];
    $('#tableCount').textContent = `${mentions.length} record${mentions.length === 1 ? '' : 's'}`;
    renderPager($('#tablePager'), tablePage, Math.ceil(mentions.length / PAGE_SIZE), (page) => { tablePage = page; renderTable(); });
    mentions.slice(tablePage * PAGE_SIZE, (tablePage + 1) * PAGE_SIZE).forEach((mention) => {
      const location = mention.location || {};
      const row = document.createElement('tr'); row.dataset.mentionId = mention.id;
      row.addEventListener('click', () => selectMention(mention.id, true));
      const mentionCell = document.createElement('td'); mentionCell.className = 'mention-label'; mentionCell.textContent = mention.mention || '—';
      const entityCell = document.createElement('td');
      const entityName = document.createElement('div'); entityName.className = 'entity-name'; entityName.textContent = location.name || 'Unresolved';
      entityCell.append(entityName);
      if (location.url) { const link = document.createElement('a'); link.className = 'entity-link'; link.href = location.url; link.target = '_blank'; link.rel = 'noopener'; link.textContent = location.url; link.addEventListener('click', (e) => e.stopPropagation()); entityCell.append(link); }
      const typeCell = document.createElement('td'); typeCell.textContent = location.type || 'Unknown';
      const mapCell = document.createElement('td'); const mapped = location.latitude !== null && location.latitude !== undefined && location.longitude !== null && location.longitude !== undefined;
      const mapTag = document.createElement('span'); mapTag.className = mapped ? 'mapped-tag' : 'unmapped-tag'; mapTag.textContent = mapped ? '● On map' : '— No coords'; mapCell.append(mapTag);
      row.append(mentionCell, entityCell, typeCell, mapCell); tbody.append(row);
    });
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  }
  function getMention(id) { return (currentResults?.mentions || []).find((m) => m.id === id); }
  function getLocation(id) { return (currentResults?.locations || []).find((l) => l.id === id); }
  function selectMention(id, center) {
    const mention = getMention(id); if (!mention) return;
    selectedMentionId = id;
    // Bring the mention's text page and register page into view first.
    const wantedText = pageOfParagraph(mention.paragraph_index);
    if (wantedText !== textPage) setTextPage(wantedText);
    const wantedTable = Math.floor(currentResults.mentions.indexOf(mention) / PAGE_SIZE);
    if (wantedTable !== tablePage) { tablePage = wantedTable; renderTable(); }
    const loc = mention.location || {};
    selectedLocationId = loc.id || null;
    document.querySelectorAll('.place-token').forEach((el) => el.classList.toggle('active', el.dataset.mentionId === id));
    document.querySelectorAll('#mentionRows tr').forEach((el) => el.classList.toggle('active-row', el.dataset.mentionId === id));
    if (center) {
      const token = document.querySelector(`.place-token[data-mention-id="${CSS.escape(id)}"]`);
      if (token) token.scrollIntoView({behavior: 'smooth', block: 'center'});
    }
    const location = getLocation(loc.id) || loc;
    if (currentEditMode === 'modify') { showEditor('modify', mention, location); return; }
    if (currentEditMode === 'delete') { showEditor('delete', mention, location); return; }
    if (currentEditMode === 'add') {
      renderActionPrompt('add', 'Add mode is active. Select a span of source text in the left panel to create a new mention.');
      return;
    }
    renderLocationDetail(location, mention);
    const marker = markersByLocation.get(location.id);
    if (marker && center && map) map.flyTo(marker.getLatLng(), Math.max(map.getZoom(), 7), {duration: .6});
    if (marker && center) marker.openPopup();
  }

  function makePopup(location, index) {
    const coords = location.latitude != null && location.longitude != null ? `${Number(location.latitude).toFixed(4)}, ${Number(location.longitude).toFixed(4)}` : 'Coordinates unavailable';
    return `<div><b>${escapeHtml(location.name)}</b><br><span>${escapeHtml(location.type || 'Unclassified place')}</span><br><small>${escapeHtml(coords)}</small></div>`;
  }

  function initMap() {
    markersByLocation = new Map();
    if (map) { map.remove(); map = null; }
    $('#map').replaceChildren();
    if (!window.L) {
      $('#map').innerHTML = '<div class="map-fallback">Leaflet did not load. Check the browser connection to unpkg.com and reload this page.</div>';
      return;
    }
    map = L.map('map', {zoomControl: true, scrollWheelZoom: true, preferCanvas: true}).setView([37.7, 22.2], 5);
    mapLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'
    }).addTo(map);
    // Only the places mentioned on the current text page get a pin.
    const onPage = new Set(pageMentions().map((m) => m.location?.id));
    const locs = (currentResults?.locations || []).filter((loc) => onPage.has(loc.id));
    const withCoords = locs.filter((loc) => Number.isFinite(Number(loc.latitude)) && Number.isFinite(Number(loc.longitude)) && loc.latitude !== null && loc.longitude !== null);
    const bounds = [];
    withCoords.forEach((location, index) => {
      const icon = L.divIcon({className: '', html: `<div class="custom-marker"><span>${index + 1}</span></div>`, iconSize: [29, 29], iconAnchor: [14, 28], popupAnchor: [0, -27]});
      const marker = L.marker([Number(location.latitude), Number(location.longitude)], {icon, title: location.name || location.modern_location || (location.kb_id ? `ToposText ${location.kb_id}` : 'Unresolved place')}).addTo(map);
      marker.bindPopup(makePopup(location, index));
      marker.on('click', () => {
        selectedLocationId = location.id;
        const mention = (currentResults.mentions || []).find((m) => m.location?.id === location.id);
        if (mention) selectMention(mention.id, true); else renderLocationDetail(location, null);
      });
      markersByLocation.set(location.id, marker);
      bounds.push([Number(location.latitude), Number(location.longitude)]);
    });
    if (bounds.length === 1) map.setView(bounds[0], 7);
    else if (bounds.length > 1) map.fitBounds(bounds, {padding: [28, 28], maxZoom: 8});
    $('#map').querySelector('.map-loading')?.remove();
    $('#mappedCountLabel').textContent = `${withCoords.length} mapped${textPages.length > 1 ? ' on this page' : ''}`;
    if (!withCoords.length) {
      const notice = document.createElement('div'); notice.className = 'map-notice';
      notice.innerHTML = '<b>No coordinates were available for the selected entities.</b><span>Links and entity matches are still available below. Add the ToposText metadata files or allow page enrichment to obtain more coordinates.</span>';
      $('#map').append(notice);
    }
    setTimeout(() => map?.invalidateSize(), 100);
  }

  function refreshMapTiles() {
    if (!map) return;
    // Leaflet basemap is neutral in both themes; refreshing forces it to redraw after layout changes.
    map.invalidateSize();
  }
  $('#resetMap').addEventListener('click', () => {
    if (!map) return;
    const points = [...markersByLocation.values()].map((marker) => marker.getLatLng());
    if (points.length === 1) map.setView(points[0], 7);
    else if (points.length > 1) map.fitBounds(points, {padding: [28, 28], maxZoom: 8});
    else map.setView([37.7, 22.2], 5);
    selectedMentionId = null; selectedLocationId = null;
    document.querySelectorAll('.place-token').forEach((el) => el.classList.remove('active'));
    document.querySelectorAll('#mentionRows tr').forEach((el) => el.classList.remove('active-row'));
    if (currentEditMode) renderActionPrompt(currentEditMode);
    else renderLocationDetail(null, null);
  });
  $('#clearSelection').addEventListener('click', () => {
    selectedMentionId = null; selectedLocationId = null;
    document.querySelectorAll('.place-token').forEach((el) => el.classList.remove('active'));
    document.querySelectorAll('#mentionRows tr').forEach((el) => el.classList.remove('active-row'));
    if (currentEditMode) renderActionPrompt(currentEditMode);
    else renderLocationDetail(null, null);
  });

  function renderLocationDetail(location, mention) {
    const panel = $('#locationDetail'); panel.replaceChildren();
    if (!location) {
      const empty = document.createElement('div'); empty.className = 'detail-empty';
      empty.innerHTML = '<div class="detail-pin">⌖</div><div><b>Select a map pin or mention</b><p>Coordinates, knowledge-base link, type and text context will appear here.</p></div>';
      panel.append(empty); return;
    }
    const coordsAvailable = location.latitude !== null && location.latitude !== undefined && location.longitude !== null && location.longitude !== undefined && Number.isFinite(Number(location.latitude)) && Number.isFinite(Number(location.longitude));
    const top = document.createElement('div'); top.className = 'detail-top';
    const titleWrap = document.createElement('div'); titleWrap.className = 'detail-title-wrap';
    const title = document.createElement('h4'); title.className = 'detail-title'; title.textContent = location.name || 'Unresolved place';
    const type = document.createElement('span'); type.className = 'type-pill'; type.textContent = location.type || 'Unclassified place';
    titleWrap.append(title, type);
    const number = document.createElement('div'); number.className = 'detail-number'; number.textContent = mention ? (mention.detection_method || 'Place detected') : `${location.mention_count || 0} mention${location.mention_count === 1 ? '' : 's'}`;
    top.append(titleWrap, number); panel.append(top);
    if (location.description) { const description = document.createElement('p'); description.className = 'detail-description'; description.textContent = location.description; panel.append(description); }
    const fields = document.createElement('div'); fields.className = 'detail-fields';
    const field = (label, value) => { const box = document.createElement('div'); box.className = 'detail-field'; const small = document.createElement('small'); small.textContent = label; const strong = document.createElement('b'); strong.textContent = value; box.append(small, strong); fields.append(box); };
    field('Latitude', coordsAvailable ? Number(location.latitude).toFixed(6) : 'Not available');
    field('Longitude', coordsAvailable ? Number(location.longitude).toFixed(6) : 'Not available');
    field('Coordinates (WGS 84)', coordsAvailable ? `${Number(location.latitude).toFixed(6)}, ${Number(location.longitude).toFixed(6)}` : '—');
    field('Knowledge-base ID', location.kb_id || '-');
    field('Wikipedia ID', location.wikipedia_id || '-');
    field('Pleiades ID', location.pleiades_id || '-');
    field('Wikidata ID', location.wikidata_id || '-');
    if (location.region) field('Region', location.region);
    if (location.country) field('Country', location.country);
    if (location.coordinate_confidence) field('KB coordinate confidence', location.coordinate_confidence);
    if (location.modern_location) field('Modern location', location.modern_location);
    panel.append(fields);
    if (location.url) { const link = document.createElement('a'); link.className = 'kb-link'; link.href = location.url; link.target = '_blank'; link.rel = 'noopener'; link.textContent = 'Open ToposText record ↗'; panel.append(link); }
    if (mention) {
      const snippet = document.createElement('div'); snippet.className = 'context-quote';
      const raw = String(mention.snippet || mention.context || '');
      snippet.textContent = `Source passage: “${raw}”`;
      panel.append(snippet);
    }
  }

  function hasCoordinates(location) {
    if (!location || location.latitude === null || location.latitude === undefined || location.longitude === null || location.longitude === undefined) return false;
    const latitude = Number(location.latitude), longitude = Number(location.longitude);
    return Number.isFinite(latitude) && Number.isFinite(longitude) && latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180;
  }

  function setEditModeButtons() {
    const ids = {add: '#addModeButton', modify: '#modifyModeButton', delete: '#deleteModeButton'};
    Object.entries(ids).forEach(([mode, selector]) => {
      const button = $(selector);
      if (button) {
        button.classList.toggle('active-mode', currentEditMode === mode);
        button.setAttribute('aria-pressed', currentEditMode === mode ? 'true' : 'false');
      }
    });
  }

  function activateEditMode(mode) {
    currentEditMode = mode;
    pendingSelection = null;
    setEditModeButtons();
    if (mode === 'modify' && selectedMentionId && getMention(selectedMentionId)) {
      selectMention(selectedMentionId, false);
    } else if (mode === 'delete' && selectedMentionId && getMention(selectedMentionId)) {
      selectMention(selectedMentionId, false);
    } else {
      renderActionPrompt(mode);
    }
  }

  function renderActionPrompt(mode, customMessage = '') {
    const panel = $('#locationDetail');
    panel.replaceChildren();
    const wrap = document.createElement('div'); wrap.className = 'edit-prompt';
    const kicker = document.createElement('div'); kicker.className = 'editor-kicker';
    kicker.textContent = mode === 'add' ? 'ADD A PLACE MENTION' : mode === 'modify' ? 'MODIFY A PLACE MENTION' : mode === 'delete' ? 'DELETE A PLACE MENTION' : 'PLACE DETAILS';
    const title = document.createElement('b');
    title.textContent = mode === 'add' ? 'Select text in the source' : mode === 'modify' ? 'Choose a highlighted mention' : mode === 'delete' ? 'Choose a highlighted mention' : 'Select a place';
    const description = document.createElement('p');
    description.textContent = customMessage || (mode === 'add'
      ? 'Click and drag over the place name in the source text. The editing card will open here. The original document text will not be changed.'
      : mode === 'modify'
        ? 'Click a highlighted mention in the source text, a row in the register, or a map pin. You can edit the place fields for that mention.'
        : mode === 'delete'
          ? 'Select a highlighted mention, then remove its annotation. The word stays in the source text and will simply no longer be highlighted.'
          : 'Choose a map pin or highlighted mention to inspect its place record.');
    wrap.append(kicker, title, description);
    if (mode) {
      const cancel = document.createElement('button'); cancel.type = 'button'; cancel.className = 'editor-cancel secondary-button'; cancel.textContent = 'Cancel mode';
      cancel.addEventListener('click', () => { currentEditMode = null; pendingSelection = null; setEditModeButtons(); renderLocationDetail(null, null); });
      wrap.append(cancel);
    }
    panel.append(wrap);
  }

  function captureSourceSelection() {
    if (currentEditMode !== 'add' || !currentResults) return;
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount) return;
    const range = selection.getRangeAt(0);
    const sourceStart = range.startContainer.nodeType === Node.ELEMENT_NODE ? range.startContainer : range.startContainer.parentElement;
    const sourceEnd = range.endContainer.nodeType === Node.ELEMENT_NODE ? range.endContainer : range.endContainer.parentElement;
    const startSection = sourceStart?.closest?.('.source-paragraph');
    const endSection = sourceEnd?.closest?.('.source-paragraph');
    if (!startSection || startSection !== endSection || !$('#sourceText').contains(startSection)) return;
    const paragraphIndex = Number(startSection.id.replace('paragraph-', ''));
    const paragraph = (currentResults.paragraphs || [])[paragraphIndex];
    if (!paragraph) return;
    let start = sourceOffset(startSection, range.startContainer, range.startOffset);
    let end = sourceOffset(startSection, range.endContainer, range.endOffset);
    if (start === null || end === null) return;
    const sourceText = String(paragraph.text || '');
    while (start < end && /\s/.test(sourceText[start] || '')) start += 1;
    while (end > start && /\s/.test(sourceText[end - 1] || '')) end -= 1;
    if (end <= start) return;
    const selectedText = sourceText.slice(start, end);
    if (!selectedText.trim()) return;
    const overlaps = (currentResults.mentions || []).some((mention) => mention.paragraph_index === paragraphIndex && start < mention.end && end > mention.start);
    if (overlaps) {
      renderActionPrompt('add', 'That selection overlaps an existing mention. Select an unannotated text span, or switch to Modify to edit the existing mention.');
      return;
    }
    pendingSelection = {paragraph_index: paragraphIndex, start, end, mention: selectedText};
    showEditor('add', null, {}, pendingSelection);
  }

  function sourceOffset(section, container, offset) {
    const number = section.querySelector('.paragraph-number');
    if (!number) return null;
    try {
      const range = document.createRange();
      range.setStartAfter(number);
      range.setEnd(container, offset);
      return range.toString().length;
    } catch (_) {
      return null;
    }
  }

  function showEditor(mode, mention, location, selection = null) {
    const panel = $('#locationDetail'); panel.replaceChildren();
    const wrap = document.createElement('div'); wrap.className = 'entity-editor';
    const header = document.createElement('div'); header.className = 'editor-heading';
    const intro = document.createElement('div');
    const kicker = document.createElement('div'); kicker.className = 'editor-kicker';
    kicker.textContent = mode === 'add' ? 'NEW MENTION' : mode === 'modify' ? 'EDIT LINKED PLACE' : 'REMOVE ANNOTATION';
    const heading = document.createElement('h4'); heading.textContent = mode === 'add' ? 'Add a place mention' : mode === 'modify' ? 'Modify this mention' : 'Delete this mention?';
    intro.append(kicker, heading); header.append(intro); wrap.append(header);

    const sourceMention = mode === 'add' ? selection?.mention : mention?.mention;
    if (mode === 'delete') {
      const warning = document.createElement('p'); warning.className = 'editor-explainer';
      warning.textContent = `Remove the annotation for “${sourceMention || ''}”? The source word will stay exactly where it is; only the geotagging record and highlight will be removed.`;
      const entity = document.createElement('div'); entity.className = 'editor-current-entity';
      entity.textContent = `${location?.name || 'Unresolved place'}${location?.kb_id ? ` · ToposText ID ${location.kb_id}` : ''}`;
      const actions = document.createElement('div'); actions.className = 'editor-actions';
      const cancel = document.createElement('button'); cancel.type = 'button'; cancel.className = 'secondary-button'; cancel.textContent = 'Keep mention';
      cancel.addEventListener('click', () => { currentEditMode = null; setEditModeButtons(); renderLocationDetail(location, mention); });
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'danger-button'; remove.textContent = 'Delete annotation';
      remove.addEventListener('click', () => deleteMention(mention.id));
      actions.append(cancel, remove); wrap.append(warning, entity, actions); panel.append(wrap); return;
    }

    const explainer = document.createElement('p'); explainer.className = 'editor-explainer';
    explainer.textContent = mode === 'add'
      ? 'The selected source text is immutable. Fill in the place metadata manually. Changing the ToposText ID will not load or overwrite any other fields.'
      : 'Only this mention’s linked place record is edited. The source wording will remain unchanged.';
    wrap.append(explainer);
    const mentionBox = document.createElement('div'); mentionBox.className = 'selected-source-mention';
    const mentionLabel = document.createElement('small'); mentionLabel.textContent = 'SOURCE TEXT · NOT EDITED';
    const mentionValue = document.createElement('b'); mentionValue.textContent = sourceMention || 'No source mention';
    mentionBox.append(mentionLabel, mentionValue); wrap.append(mentionBox);

    const form = document.createElement('form'); form.className = 'editor-form';
    const fields = document.createElement('div'); fields.className = 'editor-fields';
    const inputs = {};
    const createField = (key, label, value, options = {}) => {
      const labelEl = document.createElement('label'); labelEl.className = `editor-field${options.wide ? ' editor-field-wide' : ''}`;
      const labelText = document.createElement('span'); labelText.textContent = label; labelEl.append(labelText);
      let input;
      if (options.multiline) { input = document.createElement('textarea'); input.rows = 3; }
      else { input = document.createElement('input'); input.type = options.number ? 'number' : 'text'; if (options.number) { input.step = 'any'; input.inputMode = 'decimal'; } }
      input.name = key; input.id = `editor-${key}`; input.value = value === null || value === undefined ? '' : String(value);
      if (options.placeholder) input.placeholder = options.placeholder;
      labelEl.append(input); fields.append(labelEl); inputs[key] = input; return input;
    };
    const base = location || {};
    createField('kb_id', 'ToposText ID', base.kb_id || '', {placeholder: 'e.g. 12345', wide: true});
    createField('name', 'Place name', mode === 'add' ? (base.name || '') : (base.name || sourceMention || ''), {wide: true});
    createField('description', 'Description', base.description || '', {multiline: true, wide: true});
    createField('type', 'Settlement / place type', base.type || 'Unclassified place', {placeholder: 'City, settlement, temple…'});
    createField('modern_location', 'Modern settlement / location', base.modern_location || '');
    createField('latitude', 'Latitude', base.latitude ?? '', {number: true, placeholder: '−90 to 90'});
    createField('longitude', 'Longitude', base.longitude ?? '', {number: true, placeholder: '−180 to 180'});
    createField('region', 'Region', base.region || '');
    createField('country', 'Country', base.country || '');
    createField('department', 'District / department', base.department || '');
    createField('url', 'Knowledge-base URL', base.url || '', {wide: true, placeholder: 'https://topostext.org/place/…'});
    createField('wikipedia_url', 'Wikipedia URL', base.wikipedia_url || '', {wide: true, placeholder: 'Optional article URL'});
    form.append(fields);
    const editorNote = document.createElement('div'); editorNote.className = 'editor-manual-note';
    editorNote.textContent = 'All place fields are entered manually. Changing the ID does not fetch or populate metadata.';
    form.append(editorNote);
    const actions = document.createElement('div'); actions.className = 'editor-actions';
    const cancel = document.createElement('button'); cancel.type = 'button'; cancel.className = 'secondary-button'; cancel.textContent = 'Cancel';
    cancel.addEventListener('click', () => { currentEditMode = null; pendingSelection = null; setEditModeButtons(); renderLocationDetail(location, mention); });
    const save = document.createElement('button'); save.type = 'submit'; save.className = 'primary-button'; save.textContent = mode === 'add' ? 'Add mention & save' : 'Save changes';
    actions.append(cancel, save); form.append(actions); wrap.append(form); panel.append(wrap);

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      try {
        const editedLocation = collectEditorLocation(inputs, sourceMention, location);
        if (mode === 'add') await addMention(selection, editedLocation);
        else await modifyMention(mention, editedLocation);
      } catch (error) {
        editorNote.textContent = error.message || 'Could not save the edit.';
        editorNote.classList.add('editor-error');
      }
    });
  }

  function collectEditorLocation(inputs, sourceMention, existingLocation) {
    const value = (key) => inputs[key]?.value.trim() || '';
    const kbId = value('kb_id');
    let latitude = value('latitude') === '' ? null : Number(value('latitude'));
    let longitude = value('longitude') === '' ? null : Number(value('longitude'));
    if ((latitude === null) !== (longitude === null)) throw new Error('Enter both latitude and longitude, or leave both blank.');
    if (latitude !== null && (!Number.isFinite(latitude) || latitude < -90 || latitude > 90)) throw new Error('Latitude must be between −90 and 90.');
    if (longitude !== null && (!Number.isFinite(longitude) || longitude < -180 || longitude > 180)) throw new Error('Longitude must be between −180 and 180.');
    const name = value('name') || (existingLocation ? existingLocation.name : '') || '';
    if (!name) throw new Error('Enter a place name before saving.');
    let url = value('url');
    if (url && !/^https?:\/\//i.test(url)) throw new Error('Knowledge-base URL must start with http:// or https://.');
    const wikipediaUrl = value('wikipedia_url');
    if (wikipediaUrl && !/^https?:\/\//i.test(wikipediaUrl)) throw new Error('Wikipedia URL must start with http:// or https://.');
    return {
      id: existingLocation?.id || `place-manual-${makeShortId()}`,
      kb_id: kbId, name, description: value('description'), type: value('type') || 'Unclassified place',
      latitude, longitude, url, region: value('region'), country: value('country'), department: value('department'),
      modern_location: value('modern_location'), wikipedia_url: wikipediaUrl,
      wikipedia_id: existingLocation?.wikipedia_id || '',
      pleiades_id: existingLocation?.pleiades_id || '',
      wikidata_id: existingLocation?.wikidata_id || '',
      toposid: kbId || existingLocation?.toposid || '',
      coordinate_confidence: existingLocation?.coordinate_confidence || '', match_method: 'manual review', mention_count: 0,
    };
  }

  function makeShortId() {
    if (window.crypto?.randomUUID) return window.crypto.randomUUID().replace(/-/g, '').slice(0, 12);
    return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  }

  async function addMention(selection, location) {
    if (!selection) throw new Error('Select some source text first.');
    const previous = JSON.stringify(currentResults);
    const paragraph = (currentResults.paragraphs || [])[selection.paragraph_index];
    if (!paragraph || String(paragraph.text || '').slice(selection.start, selection.end) !== selection.mention) throw new Error('The selected source text changed. Please select it again.');
    if ((currentResults.mentions || []).some((item) => item.paragraph_index === selection.paragraph_index && selection.start < item.end && selection.end > item.start)) throw new Error('This selection overlaps an existing place mention.');
    location.id = `place-manual-${makeShortId()}`;
    const sourceText = String(paragraph.text || '');
    const mention = {
      id: `m-manual-${selection.paragraph_index}-${selection.start}-${makeShortId()}`,
      paragraph_index: selection.paragraph_index, start: selection.start, end: selection.end,
      mention: selection.mention, label: 'PLACE', recontext: '', match_score: null,
      detection_method: 'Manual addition', match_method: 'manual', context: sourceText,
      snippet: sourceText.slice(Math.max(0, selection.start - 80), Math.min(sourceText.length, selection.end + 80)),
      location,
    };
    currentResults.mentions.push(mention);
    currentResults.mentions.sort((a, b) => a.paragraph_index - b.paragraph_index || a.start - b.start);
    currentResults.locations = currentResults.locations || [];
    currentResults.locations.push(location);
    await persistReview(previous);
  }

  async function modifyMention(mention, editedLocation) {
    if (!mention) throw new Error('Select a mention to modify.');
    const previous = JSON.stringify(currentResults);
    const original = mention.location || {};
    const linkedCount = (currentResults.mentions || []).filter((item) => item.location?.id === original.id).length;
    const newLocationId = linkedCount > 1 ? `place-edited-${makeShortId()}` : (original.id || `place-edited-${makeShortId()}`);
    const updatedLocation = {...editedLocation, id: newLocationId};
    mention.location = updatedLocation;
    currentResults.locations = (currentResults.locations || []).filter((item) => item.id !== original.id || linkedCount > 1);
    const existingIndex = currentResults.locations.findIndex((item) => item.id === updatedLocation.id);
    if (existingIndex >= 0) currentResults.locations[existingIndex] = updatedLocation;
    else currentResults.locations.push(updatedLocation);
    await persistReview(previous);
  }

  async function deleteMention(mentionId) {
    const previous = JSON.stringify(currentResults);
    currentResults.mentions = (currentResults.mentions || []).filter((mention) => mention.id !== mentionId);
    // The original text is retained; only the mention annotation is removed.
    await persistReview(previous);
  }

  async function persistReview(previousSnapshot) {
    try {
      const response = await fetch(`/api/jobs/${currentResults.job_id}/review`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mentions: currentResults.mentions || [], locations: currentResults.locations || []}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'The review changes could not be saved.');
      currentResults = payload.result;
      currentEditMode = null; pendingSelection = null; selectedMentionId = null; selectedLocationId = null;
      setEditModeButtons();
      renderResults(currentResults);
      $('#resultsSubtitle').textContent = `${currentResults.filename} · Your review changes are saved to the JSON and TEI exports.`;
    } catch (error) {
      currentResults = JSON.parse(previousSnapshot);
      renderResults(currentResults);
      $('#resultsSubtitle').textContent = `Could not save review changes: ${error.message || 'unknown error'}. The page has been restored to the previous saved result.`;
      throw error;
    }
  }

  function scrollToElement(element, offset = 0) {
    const y = element.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({top: y, behavior: 'smooth'});
  }
})();
