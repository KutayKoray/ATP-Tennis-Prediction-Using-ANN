/* ═══════════════════════════════════════════════════════════
   Tennis Match Prediction AI — App Logic
   ═══════════════════════════════════════════════════════════ */

// ⚡ DEPLOYMENT: Empty string for Vercel (same domain), or full URL for separate backend
const API_BASE = '';

// ─── Set current month on load ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const monthSelect = document.getElementById('month-select');
    const now = new Date();
    monthSelect.value = now.getMonth() + 1;

    initPlayerSearch('player1-input', 'player1-dropdown');
    initPlayerSearch('player2-input', 'player2-dropdown');
    initStatCounters();
    initScrollAnimations();
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
                dropdown.innerHTML = '<div class="player-dropdown-empty">⚠️ Backend not connected</div>';
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
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}


// ═══════════════════════════════════════════════════════════
// QUICK MATCHUPS
// ═══════════════════════════════════════════════════════════

function fillMatchup(p1, p2, surface, tournament) {
    document.getElementById('player1-input').value = p1;
    document.getElementById('player2-input').value = p2;
    document.getElementById('surface-select').value = surface;

    // Map tournament text to value
    const tourneyMap = {
        'Grand Slam': '4', 'Masters 1000': '3', 'ATP 500': '2', 'Other': '1'
    };
    document.getElementById('tournament-select').value = tourneyMap[tournament] || '2';

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

    // High-DPI support
    const dpr = window.devicePixelRatio || 1;
    const W = 900;
    const H = 420;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);

    // Layer config: [neuronCount shown, color]
    const layers = [
        { n: 7, color: '#00e5ff', glow: 'rgba(0, 229, 255, %%)', label: '37' },     // Input
        { n: 10, color: '#a855f7', glow: 'rgba(168, 85, 247, %%)', label: '128' },   // Hidden 1
        { n: 7, color: '#a855f7', glow: 'rgba(168, 85, 247, %%)', label: '64' },     // Hidden 2
        { n: 1, color: '#39ff14', glow: 'rgba(57, 255, 20, %%)', label: '1' },       // Output
    ];

    const padding = { x: 80, y: 35 };
    const neuronRadius = 10;
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
    const MAX_PARTICLES = 35;

    function spawnParticle() {
        if (particles.length >= MAX_PARTICLES) return;
        // Pick a random connection between adjacent layers
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
            size: 2 + Math.random() * 2
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
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }
    }

    function drawNeurons(time) {
        for (let li = 0; li < layers.length; li++) {
            const layer = layers[li];
            for (let ni = 0; ni < neurons[li].length; ni++) {
                const { x, y } = neurons[li][ni];
                const pulse = 1 + 0.15 * Math.sin(time * 0.002 + ni * 0.7 + li * 1.3);

                // Outer glow
                const grad = ctx.createRadialGradient(x, y, 0, x, y, neuronRadius * 3 * pulse);
                grad.addColorStop(0, layer.glow.replace('%%', '0.25'));
                grad.addColorStop(1, layer.glow.replace('%%', '0'));
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius * 3 * pulse, 0, Math.PI * 2);
                ctx.fillStyle = grad;
                ctx.fill();

                // Neuron body
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius * pulse, 0, Math.PI * 2);
                ctx.fillStyle = layer.glow.replace('%%', '0.2');
                ctx.fill();
                ctx.strokeStyle = layer.color;
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Inner bright core
                ctx.beginPath();
                ctx.arc(x, y, neuronRadius * 0.4 * pulse, 0, Math.PI * 2);
                ctx.fillStyle = layer.color;
                ctx.fill();
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
            const alpha = Math.sin(p.t * Math.PI); // Fade in/out

            // Glowing particle trail
            const grad = ctx.createRadialGradient(x, y, 0, x, y, p.size * 4);
            grad.addColorStop(0, p.color);
            grad.addColorStop(0.5, p.color + '40');
            grad.addColorStop(1, 'transparent');
            ctx.beginPath();
            ctx.arc(x, y, p.size * 4, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.globalAlpha = alpha * 0.6;
            ctx.fill();

            // Bright core
            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.globalAlpha = alpha;
            ctx.fill();

            ctx.globalAlpha = 1;
        }
    }

    function drawDotIndicators() {
        // Show "..." dots to indicate there are more neurons
        const dotsColor = 'rgba(255, 255, 255, 0.25)';
        for (let li = 0; li < layers.length; li++) {
            if (layers[li].n < 3) continue; // No dots for output layer
            const x = layerX[li];
            const topY = neurons[li][0].y;
            const botY = neurons[li][neurons[li].length - 1].y;
            const midY = (topY + botY) / 2;

            // Ellipsis above and below the middle
            for (const offsetY of [midY - 8, midY, midY + 8]) {
                // Don't draw if too close to a neuron
                const tooClose = neurons[li].some(n => Math.abs(n.y - offsetY) < 18);
                if (!tooClose) {
                    ctx.beginPath();
                    ctx.arc(x + 22, offsetY, 1.8, 0, Math.PI * 2);
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

        // Spawn particles periodically
        spawnTimer++;
        if (spawnTimer % 3 === 0) spawnParticle();

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

// Init on load
document.addEventListener('DOMContentLoaded', () => {
    initNeuralNetwork();
});


// ═══════════════════════════════════════════════════════════
// NAVBAR SCROLL EFFECT
// ═══════════════════════════════════════════════════════════

window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.style.padding = '10px 0';
        navbar.style.background = 'rgba(10, 14, 26, 0.95)';
    } else {
        navbar.style.padding = '16px 0';
        navbar.style.background = 'rgba(10, 14, 26, 0.85)';
    }
});
