window.onscroll = function () {
    if (pageYOffset >= 500) {
        document.getElementById('backToTop').style.visibility = "visible";
    } else {
 document.getElementById('backToTop').style.visibility = "hidden";
    }
};
