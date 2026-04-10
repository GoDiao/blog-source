window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"], ['$', '$']],
        displayMath: [["\\[", "\\]"], ['$$', '$$']],
        processEscapes: true,
        processEnvironments: true,
        packages: {'[+]': ['ams']}
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    loader: {
        load: ['[tex]/ams']
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
})
