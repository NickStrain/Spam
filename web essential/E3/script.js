document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('registrationForm');
    const emailInput = document.getElementById('email');
    const emailError = document.getElementById('emailError');

    form.addEventListener('submit', function (event) {
        if (!validateEmail(emailInput.value)) {
            emailError.textContent = 'Invalid email address';
            event.preventDefault();
        } else {
            emailError.textContent = '';
        }
    });

    function validateEmail(email) {
        const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return regex.test(email);
    }
});
