function toggleFilter(button) {
    console.log('Toggle function called');
    const form = button.nextElementSibling;
    if (form && form.classList.contains('filter-form')) {
      form.style.display = 'block';
      button.style.display = 'none';
    }
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    const filterForms = document.querySelectorAll('.filter-form');
    filterForms.forEach(form => form.style.display = 'none');
  });


  
  