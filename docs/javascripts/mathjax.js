window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"], ['$', '$']],
        displayMath: [["\\[", "\\]"], ['$$', '$$']],
        processEscapes: true,
        processEnvironments: true,
        packages: {'[+]': ['ams', 'newcommand', 'configmacros']}
    },
    chtml: {
        displayAlign: 'left',
        displayIndent: '2em'
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea'],
        processHtmlClass: 'arithmatex',
        renderActions: {
            addMenu: []
        }
    },
    loader: {
        load: ['[tex]/ams', '[tex]/newcommand', '[tex]/configmacros']
    },
    startup: {
        pageReady: function() {
            return MathJax.startup.defaultPageReady().then(function() {
                // Re-typeset on MkDocs navigation (if instant loading is enabled)
                if (typeof document$ !== 'undefined') {
                    document$.subscribe(function() {
                        MathJax.typesetPromise();
                    });
                }
            });
        }
    }
};
