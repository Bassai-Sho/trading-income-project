// Trading AI — custom header injection
// Overrides Chainlit's hardcoded favicon and sets the page title
(function() {
  // Override favicon
  const existing = document.querySelector("link[rel='icon']");
  if (existing) existing.remove();
  const favicon = document.createElement("link");
  favicon.rel = "icon";
  favicon.href = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>";
  document.head.appendChild(favicon);

  // Set page title
  document.title = "Trading AI";
})();
