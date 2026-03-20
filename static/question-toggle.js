document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.question-toggle').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var content = this.closest('.research-question').querySelector('.question-content');
      var expanded = this.getAttribute('aria-expanded') === 'true';
      this.setAttribute('aria-expanded', String(!expanded));
      content.style.display = expanded ? 'none' : 'block';
      this.querySelector('i').classList.toggle('fa-chevron-down', expanded);
      this.querySelector('i').classList.toggle('fa-chevron-up', !expanded);
    });
  });
});

function initGallery(id) {
  var wrapper = document.getElementById(id);
  if (!wrapper) return;
  var slides = wrapper.querySelectorAll('.gallery-slide');
  var prev = wrapper.querySelector('.gallery-prev');
  var next = wrapper.querySelector('.gallery-next');
  var counter = wrapper.querySelector('.gallery-counter');
  var idx = 0;

  function show(i) {
    slides.forEach(function (s) { s.classList.remove('active'); });
    idx = (i + slides.length) % slides.length;
    slides[idx].classList.add('active');
    if (counter) counter.textContent = (idx + 1) + ' / ' + slides.length;
  }

  if (prev) prev.addEventListener('click', function () { show(idx - 1); });
  if (next) next.addEventListener('click', function () { show(idx + 1); });
}

function copyCitation() {
  var text = document.getElementById('bibtex-content').textContent;
  navigator.clipboard.writeText(text).then(function () {
    var btn = document.querySelector('.copy-btn');
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    setTimeout(function () {
      btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
    }, 2000);
  });
}
