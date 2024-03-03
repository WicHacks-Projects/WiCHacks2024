window.addEventListener('scroll', function() {
    var scrollPosition = window.scrollY;
    var container = document.querySelector('.container');
    container.style.backgroundPositionY = -scrollPosition + 'px';
});
