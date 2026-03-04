/* ═══════════════════════════════════════════════════════════
   Tennis Match Prediction AI — App Logic
   ═══════════════════════════════════════════════════════════ */

// ⚡ DEPLOYMENT: Empty string for Vercel (same domain), or full URL for separate backend
const API_BASE = '';

// ─── Set current month on load ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initPlayerSearch('player1-input', 'player1-dropdown');
    initPlayerSearch('player2-input', 'player2-dropdown');
    initStatCounters();
    initScrollAnimations();
    initNeuralNetwork();
});

// ═══════════════════════════════════════════════════════════
// PLAYER SEARCH (Searchable Dropdown)
// ═══════════════════════════════════════════════════════════

let searchTimeout = null;

function initPlayerSearch(inputId, dropdownId) {
    const input = document.getElementById(inputId);
    const dropdown = document.getElementById(dropdownId);

    input.addEventListener('input', () => {
        const query = input.value.trim();
        clearTimeout(searchTimeout);

        if (query.length < 2) {
            dropdown.classList.remove('show');
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const res = await fetch(`${API_BASE}/api/players?q=${encodeURIComponent(query)}&limit=15`);
                const data = await res.json();

                if (data.players && data.players.length > 0) {
                    dropdown.innerHTML = data.players.map(name =>
                        `<div class="player-dropdown-item" onclick="selectPlayer('${inputId}', '${dropdownId}', '${escapeHtml(name)}')">${highlightMatch(name, query)}</div>`
                    ).join('');
                    dropdown.classList.add('show');
                } else {
                    dropdown.innerHTML = '<div class="player-dropdown-empty">No players found</div>';
                    dropdown.classList.add('show');
                }
            } catch (err) {
                dropdown.innerHTML = '<div class="player-dropdown-empty">⚠️ Server is warming up — please try again in a few seconds</div>';
                dropdown.classList.add('show');
            }
        }, 250);
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.remove('show');
        }
    });

    // Keyboard navigation
    input.addEventListener('keydown', (e) => {
        const items = dropdown.querySelectorAll('.player-dropdown-item');
        const active = dropdown.querySelector('.player-dropdown-item.active');
        let idx = Array.from(items).indexOf(active);

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (active) active.classList.remove('active');
            idx = (idx + 1) % items.length;
            items[idx]?.classList.add('active');
            items[idx]?.scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (active) active.classList.remove('active');
            idx = idx <= 0 ? items.length - 1 : idx - 1;
            items[idx]?.classList.add('active');
            items[idx]?.scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (active) {
                active.click();
            } else if (items.length > 0) {
                items[0].click();
            }
        } else if (e.key === 'Escape') {
            dropdown.classList.remove('show');
        }
    });
}

function selectPlayer(inputId, dropdownId, name) {
    document.getElementById(inputId).value = name;
    document.getElementById(dropdownId).classList.remove('show');
}

function highlightMatch(text, query) {
    const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
    return text.replace(regex, '<strong style="color:var(--accent-cyan)">$1</strong>');
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/'/g, "\\'")
        .replace(/"/g, '&quot;');
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}


// ═══════════════════════════════════════════════════════════
// QUICK MATCHUPS
// ═══════════════════════════════════════════════════════════

function fillMatchup(p1, p2, surface) {
    document.getElementById('player1-input').value = p1;
    document.getElementById('player2-input').value = p2;
    document.getElementById('surface-select').value = surface;

    // Scroll to predict section smoothly
    document.getElementById('predict').scrollIntoView({ behavior: 'smooth', block: 'center' });
}


// ═══════════════════════════════════════════════════════════
// PREDICTION
// ═══════════════════════════════════════════════════════════

async function predictMatch() {
    const player1 = document.getElementById('player1-input').value.trim();
    const player2 = document.getElementById('player2-input').value.trim();
    const surface = document.getElementById('surface-select').value;
    const errorEl = document.getElementById('error-message');
    const resultEl = document.getElementById('prediction-result');
    const btn = document.getElementById('predict-btn');

    // Hide previous results/errors
    errorEl.classList.remove('show');
    resultEl.classList.remove('show');

    // Validate
    if (!player1 || !player2) {
        showError('Please select both players');
        return;
    }
    if (player1.toLowerCase() === player2.toLowerCase()) {
        showError('Please select two different players');
        return;
    }

    // Loading state
    btn.classList.add('loading');

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player1, player2, surface })
        });

        const data = await res.json();

        if (!res.ok) {
            showError(data.error || 'Prediction failed');
            btn.classList.remove('loading');
            return;
        }

        displayResult(data);
    } catch (err) {
        showError('⚠️ Could not connect to the prediction API. Please try again in a moment.');
    }

    btn.classList.remove('loading');
}

function showError(msg) {
    const el = document.getElementById('error-message');
    el.textContent = msg;
    el.classList.add('show');
}


// ═══════════════════════════════════════════════════════════
// DISPLAY RESULT
// ═══════════════════════════════════════════════════════════

function displayResult(data) {
    const resultEl = document.getElementById('prediction-result');

    // Winner name + confidence
    document.getElementById('result-winner-name').textContent = data.predicted_winner;
    document.getElementById('result-confidence').textContent = data.confidence.toFixed(1) + '%';

    // Probability bar
    const p1Pct = (data.player1.win_probability * 100).toFixed(1);
    const p2Pct = (data.player2.win_probability * 100).toFixed(1);

    document.getElementById('prob-p1-name').textContent = data.player1.name;
    document.getElementById('prob-p2-name').textContent = data.player2.name;
    document.getElementById('prob-p1-pct').textContent = p1Pct + '%';
    document.getElementById('prob-p2-pct').textContent = p2Pct + '%';

    // Set winner/loser label styling
    const p1Label = document.getElementById('prob-p1-label');
    const p2Label = document.getElementById('prob-p2-label');
    p1Label.className = 'prob-label ' + (data.predicted_winner === data.player1.name ? 'winner' : 'loser');
    p2Label.className = 'prob-label ' + (data.predicted_winner === data.player2.name ? 'winner' : 'loser');
    p2Label.style.textAlign = 'right';

    // Animate probability fill
    const fill = document.getElementById('probability-fill');
    fill.style.width = '50%'; // Reset
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            fill.style.width = p1Pct + '%';
        });
    });

    // H2H
    const h2h = data.h2h;
    const p1Wins = Math.round(h2h.p1_win_rate * h2h.matches);
    const p2Wins = h2h.matches - p1Wins;
    document.getElementById('h2h-value').textContent =
        h2h.matches > 0
            ? `${data.player1.name.split(' ').pop()} ${p1Wins} — ${p2Wins} ${data.player2.name.split(' ').pop()}  (${h2h.matches} matches)`
            : 'No previous encounters';

    // Player stat cards
    renderPlayerStats('result-p1', data.player1, data.predicted_winner === data.player1.name);
    renderPlayerStats('result-p2', data.player2, data.predicted_winner === data.player2.name);

    // Show result
    resultEl.classList.add('show');
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function renderPlayerStats(prefix, player, isWinner) {
    const card = document.getElementById(prefix + '-card');
    const nameEl = document.getElementById(prefix + '-name');
    const statsEl = document.getElementById(prefix + '-stats');

    // Winner styling
    if (isWinner) {
        card.classList.add('winner');
        nameEl.innerHTML = `${player.name} <span class="winner-badge">WINNER</span>`;
    } else {
        card.classList.remove('winner');
        nameEl.textContent = player.name;
    }

    const s = player.stats;
    statsEl.innerHTML = `
    <div class="stat-row"><span class="stat-name">Rank Points</span><span class="stat-val">${s.rank_points.toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-name">Age</span><span class="stat-val">${s.age}</span></div>
    <div class="stat-row"><span class="stat-name">Height</span><span class="stat-val">${s.height} cm</span></div>
    <div class="stat-row"><span class="stat-name">Hand</span><span class="stat-val">${s.hand}</span></div>
    <div class="stat-row"><span class="stat-name">Career Win Rate</span><span class="stat-val">${s.career_win_rate}%</span></div>
    <div class="stat-row"><span class="stat-name">Surface Win Rate</span><span class="stat-val">${s.surface_win_rate}%</span></div>
    <div class="stat-row"><span class="stat-name">Recent Form</span><span class="stat-val">${s.recent_form}%</span></div>
    <div class="stat-row"><span class="stat-name">Avg Ace Rate</span><span class="stat-val">${s.avg_ace_rate}%</span></div>
    <div class="stat-row"><span class="stat-name">Avg 1st Serve %</span><span class="stat-val">${s.avg_1st_serve_pct}%</span></div>
  `;
}


// ═══════════════════════════════════════════════════════════
// ANIMATED STAT COUNTERS
// ═══════════════════════════════════════════════════════════

function initStatCounters() {
    const counters = document.querySelectorAll('.stat-value[data-count]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCount(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(el => observer.observe(el));
}

function animateCount(el) {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const prefix = el.dataset.prefix || '';
    const decimals = parseInt(el.dataset.decimals ?? '1');
    const duration = 1500;
    const start = performance.now();

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;

        el.textContent = prefix + current.toFixed(decimals) + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}


// ═══════════════════════════════════════════════════════════
// SCROLL ANIMATIONS (Fade In)
// ═══════════════════════════════════════════════════════════

function initScrollAnimations() {
    const elements = document.querySelectorAll('.fade-in');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                // Stagger delay for siblings
                const siblings = entry.target.parentElement?.querySelectorAll('.fade-in');
                const idx = siblings ? Array.from(siblings).indexOf(entry.target) : 0;
                entry.target.style.transitionDelay = `${idx * 0.1}s`;
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15 });

    elements.forEach(el => observer.observe(el));
}


// ═══════════════════════════════════════════════════════════
// NEURAL NETWORK CANVAS VISUALIZATION
// ═══════════════════════════════════════════════════════════

function initNeuralNetwork() {
    const canvas = document.getElementById('nn-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const wrapper = canvas.parentElement;

    // Responsive sizing
    function getSize() {
        const containerW = wrapper.clientWidth || 900;
        const W = Math.min(containerW, 900);
        const H = Math.round(W * 0.47); // maintain aspect ratio
        return { W, H };
    }

    let { W, H } = getSize();
    const isMobile = W < 500;

    // High-DPI support
    const dpr = window.devicePixelRatio || 1;

    function setupCanvas() {
        const size = getSize();
        W = size.W;
        H = size.H;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
    }
    setupCanvas();

    // Layer config — fewer neurons on mobile for clarity
    const layers = isMobile ? [
        { n: 4, color: '#c8a951', glow: 'rgba(200, 169, 81, %%)', label: '37' },
        { n: 6, color: '#1a3a5c', glow: 'rgba(26, 58, 92, %%)', label: '128' },
        { n: 4, color: '#1a3a5c', glow: 'rgba(26, 58, 92, %%)', label: '64' },
        { n: 1, color: '#5c7a3a', glow: 'rgba(92, 122, 58, %%)', label: '1' },
    ] : [
        { n: 7, color: '#c8a951', glow: 'rgba(200, 169, 81, %%)', label: '37' },
        { n: 10, color: '#1a3a5c', glow: 'rgba(26, 58, 92, %%)', label: '128' },
        { n: 7, color: '#1a3a5c', glow: 'rgba(26, 58, 92, %%)', label: '64' },
        { n: 1, color: '#5c7a3a', glow: 'rgba(92, 122, 58, %%)', label: '1' },
    ];

    const padding = { x: isMobile ? 40 : 80, y: isMobile ? 25 : 35 };
    const neuronRadius = isMobile ? 7 : 10;
    const layerX = layers.map((_, i) => padding.x + i * ((W - padding.x * 2) / (layers.length - 1)));

    // Compute neuron Y positions for each layer
    const neurons = layers.map((layer, li) => {
        const positions = [];
        const totalH = H - padding.y * 2;
        for (let ni = 0; ni < layer.n; ni++) {
            const y = layer.n === 1
                ? H / 2
                : padding.y + (ni / (layer.n - 1)) * totalH;
            positions.push({ x: layerX[li], y });
        }
        return positions;
    });

    // Signal particles
    const particles = [];
    const MAX_PARTICLES = isMobile ? 15 : 35;

    function spawnParticle() {
        if (particles.length >= MAX_PARTICLES) return;
        const li = Math.floor(Math.random() * (layers.length - 1));
        const from = neurons[li][Math.floor(Math.random() * neurons[li].length)];
        const to = neurons[li + 1][Math.floor(Math.random() * neurons[li + 1].length)];
        const color = layers[li].color;
        particles.push({
            fromX: from.x, fromY: from.y,
            toX: to.x, toY: to.y,
            t: 0,
            speed: 0.008 + Math.random() * 0.012,
            color,
            size: isMobile ? 1.5 + Math.random() * 1.5 : 2 + Math.random() * 2
        });
    }

    // ── Draw Functions ──

    function drawConnections() {
        for (let li = 0; li < layers.length - 1; li++) {
            const fromLayer = neurons[li];
            const toLayer = neurons[li + 1];
            for (const from of fromLayer) {
                for (const to of toLayer) {
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
                    ctx.lineWidth = isMobile ? 0.5 : 0.8;
                    ctx.stroke();
                }
            }
        }
    }

    function drawNeurons(time) {
        ctx.globalAlpha = 1;
        for (let li = 0; li < layers.length; li++) {
            const layer = layers[li];
            for (let ni = 0; ni < neurons[li].length; ni++) {
                const { x, y } = neurons[li][ni];
                const size = neuronRadius * 2.2;

                // White backing circle so emoji is fully visible over connection lines
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius * 0.9, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffff';
                ctx.fill();

                ctx.font = `${size}px serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('🎾', x, y);
            }
        }
    }

    function drawParticles() {
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.t += p.speed;
            if (p.t >= 1) {
                particles.splice(i, 1);
                continue;
            }

            const x = p.fromX + (p.toX - p.fromX) * p.t;
            const y = p.fromY + (p.toY - p.fromY) * p.t;
            const alpha = Math.sin(p.t * Math.PI);

            // Soft halo
            ctx.beginPath();
            ctx.arc(x, y, p.size * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = alpha * 0.15;
            ctx.fill();

            // Solid tennis-ball core
            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = '#c4d600';
            ctx.globalAlpha = alpha * 0.9;
            ctx.fill();

            ctx.globalAlpha = 1;
        }
    }

    function drawDotIndicators() {
        const dotsColor = 'rgba(0, 0, 0, 0.2)';
        for (let li = 0; li < layers.length; li++) {
            if (layers[li].n < 3) continue;
            const x = layerX[li];
            const topY = neurons[li][0].y;
            const botY = neurons[li][neurons[li].length - 1].y;
            const midY = (topY + botY) / 2;

            for (const offsetY of [midY - 8, midY, midY + 8]) {
                const tooClose = neurons[li].some(n => Math.abs(n.y - offsetY) < 18);
                if (!tooClose) {
                    ctx.beginPath();
                    ctx.arc(x + (isMobile ? 14 : 22), offsetY, 1.8, 0, Math.PI * 2);
                    ctx.fillStyle = dotsColor;
                    ctx.fill();
                }
            }
        }
    }

    // ── Animation Loop ──
    let animFrameId;
    let spawnTimer = 0;

    function animate(time) {
        ctx.clearRect(0, 0, W, H);
        drawConnections();
        drawNeurons(time);
        spawnTimer++;
        if (spawnTimer % (isMobile ? 5 : 3) === 0) spawnParticle();
        drawParticles();
        drawDotIndicators();
        animFrameId = requestAnimationFrame(animate);
    }

    // Start only when visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (!animFrameId) animFrameId = requestAnimationFrame(animate);
            } else {
                if (animFrameId) {
                    cancelAnimationFrame(animFrameId);
                    animFrameId = null;
                }
            }
        });
    }, { threshold: 0.1 });

    observer.observe(canvas);
}

// Init neural network on load (merged into the main DOMContentLoaded at the top — line 9)
// initNeuralNetwork is called from the primary listener.


// ═══════════════════════════════════════════════════════════
// NAVBAR SCROLL EFFECT
// ═══════════════════════════════════════════════════════════

let scrollProgress = 0;

window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.style.padding = '10px 0';
        navbar.style.background = 'rgba(250, 249, 246, 0.97)';
    } else {
        navbar.style.padding = '16px 0';
        navbar.style.background = 'rgba(250, 249, 246, 0.9)';
    }

    // Court canvas scroll progress — based on wrapper scroll
    const wrapper = document.getElementById('heroWrapper');
    if (wrapper) {
        const rect = wrapper.getBoundingClientRect();
        const wrapperScroll = -rect.top; // how far into the wrapper we've scrolled
        const scrollRange = wrapper.offsetHeight - window.innerHeight; // extra scroll distance
        scrollProgress = Math.max(0, Math.min(wrapperScroll / scrollRange, 1));
    }

    // ── Hero Parallax: heading up, stats down, court reveals between ──
    const heroHeading = document.querySelector('.hero h1');
    const heroBadge = document.querySelector('.hero-badge');
    const heroSubtitle = document.querySelector('.hero-subtitle');
    const heroStats = document.querySelector('.hero-stats');

    if (heroHeading) {
        heroHeading.style.transform = `translateY(${scrollProgress * -200}px)`;
        heroHeading.style.opacity = Math.max(1 - scrollProgress * 2.5, 0);
    }
    if (heroBadge) {
        heroBadge.style.transform = `translateY(${scrollProgress * -180}px)`;
        heroBadge.style.opacity = Math.max(1 - scrollProgress * 3, 0);
    }
    if (heroSubtitle) {
        heroSubtitle.style.transform = `translateY(${scrollProgress * -160}px)`;
        heroSubtitle.style.opacity = Math.max(1 - scrollProgress * 2.5, 0);
    }
    if (heroStats) {
        heroStats.style.transform = `translateY(${scrollProgress * 150}px)`;
        heroStats.style.opacity = Math.max(1 - scrollProgress * 2, 0);
    }
});

// ═══════════════════════════════════════════════════════════
// SCROLL-DRIVEN TENNIS BALL ON COURT IMAGE
// ═══════════════════════════════════════════════════════════

function initCourtBall() {
    const canvas = document.getElementById('ballCanvas');
    const courtBg = document.getElementById('courtBg');
    if (!canvas || !courtBg) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    let W, H;

    function resize() {
        const rect = courtBg.getBoundingClientRect();
        W = rect.width;
        H = rect.height;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
    }
    resize();
    window.addEventListener('resize', resize);

    let lastProgress = -1;

    function render() {
        const p = scrollProgress; // 0 (top) → 1 (scrolled past hero)

        // Only repaint if scroll changed
        if (Math.abs(p - lastProgress) < 0.001) {
            requestAnimationFrame(render);
            return;
        }
        lastProgress = p;

        // Court zoom + move upward as we approach
        const scale = 1 + p * 0.6;
        const courtUpPercent = -30 - p * 25; // starts at -30%, moves to -55%
        courtBg.style.transform = `translateX(-50%) translateY(${courtUpPercent}%) scale(${scale})`;
        courtBg.style.opacity = 0.10 + p * 0.22;

        ctx.clearRect(0, 0, W, H);

        // Ball appears after 5% scroll
        if (p < 0.05) {
            requestAnimationFrame(render);
            return;
        }

        const ballP = Math.min((p - 0.05) / 0.90, 1);
        const t = ballP;

        // Ball trajectory: far-RIGHT corner of court → over net → near-LEFT on our half
        const startX = W * 0.65, startY = H * 0.30;
        const midX = W * 0.48, midY = H * 0.38;
        const endX = W * 0.25, endY = H * 0.72;  // between service & baseline

        // Quadratic bezier path
        const bx = (1 - t) * (1 - t) * startX + 2 * (1 - t) * t * midX + t * t * endX;
        const baseY = (1 - t) * (1 - t) * startY + 2 * (1 - t) * t * midY + t * t * endY;

        // Arc above the net
        const arcHeight = Math.sin(t * Math.PI) * (40 + t * 30);

        const ballX = bx;
        const ballY = baseY - arcHeight;

        // Ball size — smaller on mobile
        const isMobileBall = W < 500;
        const ballR = isMobileBall ? (2 + t * 3) : (3 + t * 5);

        // ── Draw ball shadow on court ──
        ctx.beginPath();
        ctx.ellipse(bx, baseY + 2, ballR * 1.4, ballR * 0.35, 0, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 0, 0, ${0.1 + t * 0.15})`;
        ctx.fill();

        // ── Draw tennis ball ──
        const grad = ctx.createRadialGradient(
            ballX - ballR * 0.3, ballY - ballR * 0.3, ballR * 0.1,
            ballX, ballY, ballR
        );
        grad.addColorStop(0, '#e0e800');
        grad.addColorStop(0.35, '#c4d600');
        grad.addColorStop(0.75, '#a0b400');
        grad.addColorStop(1, '#7a8c00');

        ctx.beginPath();
        ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        // Felt texture ring
        ctx.beginPath();
        ctx.arc(ballX, ballY, ballR * 0.92, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(180, 200, 0, 0.25)';
        ctx.lineWidth = ballR * 0.2;
        ctx.stroke();

        // Seam lines (rotate based on progress)
        ctx.save();
        ctx.beginPath();
        ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
        ctx.clip();

        ctx.translate(ballX, ballY);
        // Spin while traveling, decelerate and stop on landing
        const spinEase = t < 0.9 ? t : 0.9 + (t - 0.9) * 0.1; // slows near end
        ctx.rotate(spinEase * Math.PI * 8);

        ctx.beginPath();
        ctx.arc(-ballR * 0.4, 0, ballR * 0.6, -1.0, 1.0);
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = ballR > 6 ? 1.5 : 1;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(ballR * 0.4, 0, ballR * 0.6, Math.PI - 1.0, Math.PI + 1.0);
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = ballR > 6 ? 1.5 : 1;
        ctx.stroke();

        ctx.restore();

        // Border
        ctx.beginPath();
        ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(100, 120, 0, 0.3)';
        ctx.lineWidth = 0.6;
        ctx.stroke();

        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
}

// Init on DOM load
document.addEventListener('DOMContentLoaded', initCourtBall);


