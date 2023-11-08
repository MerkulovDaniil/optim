document.addEventListener('DOMContentLoaded', (event) => {
    const toggleButton = document.getElementById('toggleSpoilers');
    
    if (!toggleButton) {
        console.error('toggleSpoilers element not found!');
        return;
    }

    const collapseElements = document.querySelectorAll('.callout-collapse');
    
    // Function to check the visibility of eligible elements
    const checkVisibility = () => {
        // Hide button if there are no elements or no elements with data-collapse="true"
        if (collapseElements.length === 0 || !Array.from(collapseElements).some(el => el.querySelector('[data-collapse="true"]'))) {
            console.warn('No collapsible elements found!');
            toggleButton.style.display = 'none'; // Hide the button
            return false;
        }
        return true;
    }

    // Check visibility on page load
    if (!checkVisibility()) return;

    toggleButton.addEventListener('click', function() {
        const isAnyVisible = Array.from(collapseElements).some(el => {
            return el.classList.contains('show') && el.querySelector('[data-collapse="true"]');
        });

        collapseElements.forEach(el => {
            if (el.querySelector('[data-collapse="true"]')) {
                if (isAnyVisible) {
                    el.classList.remove('show');
                } else {
                    el.classList.add('show');
                }
            }
        });

        this.textContent = isAnyVisible ? '🔽' : '🔼';
    });
});
