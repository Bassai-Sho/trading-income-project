(function() {
    function applyFixes() {
        // 1. Hide the thread title "Re" text that appears next to the pencil icon
        //    It's the span inside button#chat-profiles (flex item 1 = span with profile name)
        //    The duplicate is actually the thread name shown BEFORE the profile button
        //    Target: the div.flex.items-center first child div's text nodes / spans
        const header = document.querySelector('#header');
        if (header) {
            const flexItems = header.querySelector('.flex.items-center');
            if (flexItems) {
                // First child contains pencil icon + thread title "Re"
                // We want to hide spans with short text (the truncated title)
                const firstChild = flexItems.children[0];
                if (firstChild) {
                    const spans = firstChild.querySelectorAll('span');
                    spans.forEach(span => {
                        // Hide spans that look like truncated titles (short text, not buttons)
                        if (span.textContent.trim().length > 0 && 
                            span.textContent.trim().length < 10 &&
                            !span.closest('button')) {
                            span.style.display = 'none';
                        }
                    });
                }
            }
        }

        // 2. Set favicon
        const existing = document.querySelector("link[rel='icon']");
        if (existing) existing.remove();
        const favicon = document.createElement("link");
        favicon.rel = "icon";
        favicon.href = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>";
        document.head.appendChild(favicon);
        document.title = "Trading AI";
    }

    // Run at multiple points to catch React hydration
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyFixes);
    } else {
        applyFixes();
    }
    setTimeout(applyFixes, 500);
    setTimeout(applyFixes, 1500);
    setTimeout(applyFixes, 3000);

    // Also observe DOM mutations to catch dynamic renders
    const observer = new MutationObserver(() => applyFixes());
    setTimeout(() => {
        observer.observe(document.body, { childList: true, subtree: true });
        // Disconnect after 10s to avoid performance impact
        setTimeout(() => observer.disconnect(), 10000);
    }, 1000);
})();
