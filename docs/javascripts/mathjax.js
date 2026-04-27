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
        pageReady: () => {
            return MathJax.startup.defaultPageReady().then(() => {
                if (typeof document$ !== 'undefined') {
                    document$.subscribe(() => {
                        MathJax.startup.promise.then(() => {
                            return MathJax.typesetPromise();
                        }).catch((err) => console.log('MathJax typeset error:', err));
                    });
                }
            });
        }
    }
};
