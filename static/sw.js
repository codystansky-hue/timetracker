// Minimal service worker — enables PWA installability.
// No caching: always fetches fresh from the network.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());
self.addEventListener('fetch', e => e.respondWith(fetch(e.request)));
