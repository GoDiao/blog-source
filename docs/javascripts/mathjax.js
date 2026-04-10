window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"], ['$', '$']],
        displayMath: [["\\[", "\\]"], ['$$', '$$']],
        processEscapes: true,
        processEnvironments: true,
        packages: {'[+]': ['ams']}
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        processHtmlClass: 'tex2jax_process',
        ignoreHtmlClass: 'tex2jax_ignore'
    },
    loader: {
        load: ['[tex]/ams']
    },
    startup: {
        pageReady: () => {
            return MathJax.startup.defaultPageReady().then(() => {
                console.log('MathJax initial typesetting complete');
            });
        }
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
})
