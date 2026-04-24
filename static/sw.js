// Minimal service worker — enables PWA installability.
// No caching: always fetches fresh from the network.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());
self.addEventListener('fetch', e => {
  // Let the browser handle downloads natively so Content-Disposition filenames are preserved
  if (e.request.url.includes('/download') || e.request.url.includes('/export')) return;
  e.respondWith(fetch(e.request));
});
