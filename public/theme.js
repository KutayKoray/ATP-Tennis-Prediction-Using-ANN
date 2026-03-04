// ─── Dark Mode Initialization & Toggle ─────────────────────────

// 1. Immediately apply theme to prevent FOUC (Flash of Unstyled Content)
(function initTheme() {
    const saved = localStorage.getItem('acepredict-theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
})();

// 2. Setup toggle button logic once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
        // Set initial icon
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        toggle.textContent = isDark ? '☀️' : '🌙';

        toggle.addEventListener('click', () => {
            const currentlyDark = document.documentElement.getAttribute('data-theme') === 'dark';
            if (currentlyDark) {
                document.documentElement.removeAttribute('data-theme');
                toggle.textContent = '🌙';
                localStorage.setItem('acepredict-theme', 'light');
            } else {
                document.documentElement.setAttribute('data-theme', 'dark');
                toggle.textContent = '☀️';
                localStorage.setItem('acepredict-theme', 'dark');
            }
        });
    }

    // Navbar scroll effect (applied to all pages)
    window.addEventListener('scroll', () => {
        const navbar = document.getElementById('navbar');
        if (navbar) {
            if (window.scrollY > 50) {
                navbar.classList.add('nav-scrolled');
            } else {
                navbar.classList.remove('nav-scrolled');
            }
        }
    });
});
