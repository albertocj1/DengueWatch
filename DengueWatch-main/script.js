// Load external scripts dynamically
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}
// Toggle mobile menu
function toggleMobileMenu() {
    const mobileMenu = document.querySelector('.mobile-menu');
    if (mobileMenu.classList.contains('hidden')) {
        mobileMenu.classList.remove('hidden');
        mobileMenu.classList.add('flex');
    } else {
        mobileMenu.classList.add('hidden');
        mobileMenu.classList.remove('flex');
    }
}

// Close mobile menu when clicking outside
function closeMobileMenuOnOutsideClick(event) {
    const mobileMenu = document.querySelector('.mobile-menu');
    const hamburgerButton = document.querySelector('.mobile-menu-button');
    
    if (!mobileMenu.contains(event.target) && 
        !hamburgerButton.contains(event.target) &&
        !mobileMenu.classList.contains('hidden')) {
        mobileMenu.classList.add('hidden');
        mobileMenu.classList.remove('flex');
    }
}
// Check authentication on all pages except login
if (!window.location.pathname.includes('login.html') && !sessionStorage.getItem('authenticated')) {
    window.location.href = 'login.html';
}
// logout functionality
document.getElementById('logoutBtn').addEventListener('click', function (e) {
    e.preventDefault();

    // Clear auth data
    localStorage.removeItem('user');
    localStorage.removeItem('token'); // if you use one
    sessionStorage.clear(); // optional

    // Redirect to login
    window.location.href = 'login.html';
});


// Main initialization function
async function initializeApp() {
// Load required scripts based on current page
    await Promise.all([
        loadScript('https://cdn.jsdelivr.net/npm/chart.js'),
        loadScript('https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js'),
        loadScript('https://cdn.jsdelivr.net/npm/animejs@3.2.1/lib/anime.min.js'),
        document.location.pathname.includes('admin.html') && 
            loadScript('https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js')
    ]);

    feather.replace();

    // Initialize mobile menu
    const hamburgerButtons = document.querySelectorAll('.mobile-menu-button');
    if (hamburgerButtons.length > 0) {
        hamburgerButtons.forEach(button => {
            button.addEventListener('click', toggleMobileMenu);
        });
        document.addEventListener('click', closeMobileMenuOnOutsideClick);
    }
// Initialize Vanta.js globe if element exists
    if (document.getElementById('vanta-globe')) {
        VANTA.GLOBE({
            el: "#vanta-globe",
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            color: 0x3b82f6,
            backgroundColor: 0x1e3a8a
        });
    }

    // Initialize charts if element exists
    if (document.getElementById('riskChart')) {
        const riskCtx = document.getElementById('riskChart').getContext('2d');

        // Count risks dynamically from paths
        const paths = document.querySelectorAll('.risk-path');
        let riskCounts = { Low: 0, Moderate: 0, High: 0 , VeryHigh: 0};

        paths.forEach(path => {
            const risk = path.dataset.risk; // 'Low', 'Moderate', 'High', 'VeryHigh'
            if (riskCounts[risk] !== undefined) {
                riskCounts[risk]++;
            }
        });
        const riskChart = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
                datasets: [{
                    data: [
                        riskCounts.Low,
                        riskCounts.Moderate,
                        riskCounts.High,
                        riskCounts.VeryHigh
                    ],
                    backgroundColor: ['#4ade80', '#fbbf24', '#f87171','#dc2626'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,       // Make chart responsive
                maintainAspectRatio: false, // Let container size dictate chart size
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // Animate elements
        anime({
            targets: '.risk-high',
            scale: [1, 1.1, 1],
            duration: 1500,
            loop: true,
            easing: 'easeInOutSine'
        });
    }

    // Admin page functionality
    if (document.getElementById('dropzone')) {
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('csv-upload');
        const uploadArea = document.getElementById('upload-area');
        const processingArea = document.getElementById('processing-area');
        const successMessage = document.getElementById('success-message');
        const previewTable = document.getElementById('preview-table');
        const submitBtn = document.getElementById('submit-btn');
        
        // Handle drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            dropzone.classList.add('border-blue-500');
        }

        function unhighlight() {
            dropzone.classList.remove('border-blue-500');
        }

        // Handle dropped files
        dropzone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }

        // Handle selected files
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });

        function handleFiles(files) {
            if (files.length && files[0].type === 'text/csv') {
                uploadArea.classList.add('hidden');
                processingArea.classList.remove('hidden');
                
                // Parse CSV
                Papa.parse(files[0], {
                    header: true,
                    complete: function(results) {
                        processingArea.classList.add('hidden');
                        successMessage.classList.remove('hidden');
                        
                        // Update preview table
                        updatePreviewTable(results.data);
                        
                        // Reset after 3 seconds
                        setTimeout(() => {
                            successMessage.classList.add('hidden');
                            uploadArea.classList.remove('hidden');
                        }, 3000);
                    },
                    error: function(error) {
                        console.error('Error parsing CSV:', error);
                        processingArea.classList.add('hidden');
                        uploadArea.classList.remove('hidden');
                        alert('Error parsing CSV file. Please check the format.');
                    }
                });
            } else {
                alert('Please upload a valid CSV file.');
            }
        }

        function updatePreviewTable(data) {
            if (data && data.length > 0) {
                previewTable.innerHTML = '';
                
                data.slice(0, 5).forEach(row => { // Show first 5 rows only
                    const tr = document.createElement('tr');
                    
                    ['Region', 'Cases', 'Date', 'Risk Level'].forEach(col => {
                        const td = document.createElement('td');
                        td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
                        td.textContent = row[col] || 'N/A';
                        tr.appendChild(td);
                    });
                    
                    previewTable.appendChild(tr);
                });
                
                if (data.length > 5) {
                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    td.colSpan = 4;
                    td.className = 'px-6 py-4 text-center text-sm text-gray-500';
                    td.textContent = `+ ${data.length - 5} more rows not shown`;
                    tr.appendChild(td);
                    previewTable.appendChild(tr);
                }
            }
        }

        // Submit button action
        submitBtn.addEventListener('click', function() {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
        <span class="flex items-center justify-center w-full">
            <i data-feather="loader" class="animate-spin w-4 h-4 mr-2"></i>
            Updating...
        </span>
        `;
        feather.replace();


        // Simulate API call
        setTimeout(() => {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Update Dashboard Data';

            // Create overlay div
            const overlay = document.createElement('div');
            overlay.className = 'fixed inset-0 bg-black/50 flex items-center justify-center z-50';
            overlay.innerHTML = `
                <div class="bg-white p-6 rounded-lg shadow-lg text-center animate-fade-in">
                    <p class="text-gray-800 font-medium"><strong>Dashboard data has been updated successfully!</strong></p>
                </div>
            `;
            document.body.appendChild(overlay);

            setTimeout(() => {
                overlay.remove();
            }, 1500);
        }, 2000);
    });
    }

    // Alerts page weather functionality
    if (document.getElementById('location-select')) {
        const locationSelect = document.getElementById("location-select");

        locationSelect.addEventListener("change", (e) => {
            // Remove ",PH" since WeatherAPI already recognizes Philippine cities
            const city = e.target.value.replace(",PH", "");
            const displayName = e.target.options[e.target.selectedIndex].textContent;
            const headingSpan = document.getElementById('weather-city');
            if (headingSpan) headingSpan.textContent = displayName;
            updateWeather(city);
        });

        // Initial load (default Manila)
        const initialDisplayName = locationSelect.options[locationSelect.selectedIndex].textContent;
        const headingSpan = document.getElementById('weather-city');
        if (headingSpan) headingSpan.textContent = initialDisplayName;
        updateWeather("Manila");

        // Update every 30 minutes
        setInterval(() => {
            const city = document.getElementById("location-select").value.replace(",PH", "");
            updateWeather(city);
        }, 30 * 60 * 1000);

        // Add some styling for weather blocks
        const style = document.createElement('style');
        style.textContent = `
            #past-week-weather > div, #forecast-weather > div {
                min-width: 100px;
            }
            #past-week-weather > div:hover, #forecast-weather > div:hover {
                background-color: #f8fafc;
                transform: scale(1.05);
                transition: all 0.2s ease;
            }
        `;
        document.head.appendChild(style);
    }
    // Map hover card for NCR map
    const mapContainer = document.getElementById('map-container');
    const ncrMap = document.getElementById('ncr-map');

    if (mapContainer && ncrMap) {
        // Create hover card
        const mapCard = document.createElement('div');
        mapCard.id = 'map-hover-card';
        mapCard.style.position = 'absolute';
        mapCard.style.pointerEvents = 'none';
        mapCard.style.opacity = '0';
        mapCard.style.transform = 'translateY(10px)';
        mapCard.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
        mapCard.style.background = 'white';
        mapCard.style.borderRadius = '12px';
        mapCard.style.boxShadow = '0 12px 28px rgba(0,0,0,0.25)';
        mapCard.style.padding = '12px 16px';
        mapCard.style.fontSize = '14px';
        mapCard.style.zIndex = '50';

        mapContainer.appendChild(mapCard);

        const paths = ncrMap.querySelectorAll('path');

        paths.forEach(path => {
            let originalNextSibling = null;

            path.addEventListener('mouseenter', () => {
                // Bring hovered path to top
                originalNextSibling = path.nextSibling;
                ncrMap.appendChild(path);

                // Card content (uses SVG data attributes)
                const name = path.getAttribute('data-name') || 'Unknown Area';
                const risk = path.getAttribute('data-risk') || 'Low';
                const cases = path.getAttribute('data-cases') || 'N/A';

                mapCard.innerHTML = `
                    <div style="font-weight:600;">${name}</div>
                    <div style="font-size:12px; margin-top:4px;">
                        Risk Level: <strong>${risk}</strong><br>
                        Cases: ${cases}
                    </div>
                `;

                mapCard.style.opacity = '1';
                mapCard.style.transform = 'translateY(0)';
            });

            path.addEventListener('mousemove', (e) => {
                const rect = mapContainer.getBoundingClientRect();
                mapCard.style.left = `${e.clientX - rect.left + 16}px`;
                mapCard.style.top = `${e.clientY - rect.top + 16}px`;
            });

            path.addEventListener('mouseleave', () => {
                // Restore original SVG order
                if (originalNextSibling) {
                    ncrMap.insertBefore(path, originalNextSibling);
                }

                // Hide card
                mapCard.style.opacity = '0';
                mapCard.style.transform = 'translateY(10px)';
            });
        });
    }
}

// Weather update function
function updateWeather(city = "Manila") {
    const apiKey = "003f133077684b34a7493149260701"; // Replace with your WeatherAPI key
    fetch(`https://api.weatherapi.com/v1/forecast.json?key=${apiKey}&q=${encodeURIComponent(city)}&days=14`)
        .then(response => response.json())
        .then(data => {
            // Current weather
            const temperature = Math.round(data.current.temp_c);
            const rainProbability = data.forecast.forecastday[0].day.daily_chance_of_rain + "%";

            // Risk calculation
            let dengueRisk = "Low";
            let advisory = "Normal weather conditions. Maintain regular prevention measures.";
            const rainChance = parseInt(data.forecast.forecastday[0].day.daily_chance_of_rain);

            if (rainChance > 70) {
                dengueRisk = "Critical";
                advisory = "Very high rain probability increases standing water. Expect increased mosquito activity.";
            } else if (rainChance > 50) {
                dengueRisk = "High";
                advisory = "Frequent rain expected. Check and eliminate standing water around your area.";
            } else if (rainChance > 30) {
                dengueRisk = "Medium";
                advisory = "Moderate rain probability. Stay alert and continue preventive actions.";
            }

            // Update current weather UI
            document.getElementById('rain-probability').textContent = rainProbability;
            document.getElementById('temperature').textContent = `${temperature}°C`;
            document.getElementById('dengue-risk').textContent = dengueRisk;
            document.getElementById('weather-advisory').textContent = advisory;
            document.getElementById('weather-update-time').textContent = new Date().toLocaleTimeString();

            // Generate past week (mock data since API doesn't provide historical)
            const pastWeekContainer = document.getElementById('past-week-weather');
            pastWeekContainer.innerHTML = '';
            for (let i = 6; i >= 0; i--) {
                const date = new Date();
                date.setDate(date.getDate() - i);
                const day = date.toLocaleDateString('en-US', { weekday: 'short' });
                const rainChance = Math.floor(Math.random() * 100);
                
                pastWeekContainer.innerHTML += `
                    <div class="text-center p-2 border rounded-lg">
                        <div class="font-medium">${day}</div>
                        <div class="text-sm text-gray-500">${date.getDate()}/${date.getMonth()+1}</div>
                        <div class="text-blue-500 font-medium">${rainChance}%</div>
                        <div class="text-sm">${rainChance > 50 ? '🌧️' : rainChance > 30 ? '⛅' : '☀️'}</div>
                    </div>
                `;
            }

            // Generate 14-day forecast
            const forecastContainer = document.getElementById('forecast-weather');
            forecastContainer.innerHTML = '';
            for (let i = 0; i < 14; i++) {
                const forecast = data.forecast.forecastday[i].day;
                const date = new Date(data.forecast.forecastday[i].date);
                const day = date.toLocaleDateString('en-US', { weekday: 'short' });
                
                forecastContainer.innerHTML += `
                    <div class="text-center p-2 border rounded-lg">
                        <div class="font-medium">${day}</div>
                        <div class="text-sm text-gray-500">${date.getDate()}/${date.getMonth()+1}</div>
                        <div class="text-blue-500 font-medium">${forecast.daily_chance_of_rain}%</div>
                        <div class="text-sm">${forecast.daily_chance_of_rain > 50 ? '🌧️' : forecast.daily_chance_of_rain > 30 ? '⛅' : '☀️'}</div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error("Error fetching weather:", error);
            document.getElementById('rain-probability').textContent = "N/A";
            document.getElementById('temperature').textContent = "N/A";
            document.getElementById('dengue-risk').textContent = "Unknown";
            document.getElementById('weather-advisory').textContent = "Weather data unavailable.";
            
            // Show error placeholders for forecast
            document.getElementById('past-week-weather').innerHTML = '<div class="text-center text-gray-500">Weather data unavailable</div>';
            document.getElementById('forecast-weather').innerHTML = '<div class="text-center text-gray-500">Weather data unavailable</div>';
        });
}
// Alert data and recommendations (would normally come from API)
const alertData = {
    "1": {
        title: "Quezon City Outbreak",
        location: "Quezon City",
        cases: "247 this week",
        increase: "120%",
        assessment: "This area has exceeded the epidemic threshold with a rapid increase in cases.",
        updated: "Today, 10:45 AM",
        status: "CRITICAL ALERT",
        risk: "high",
        recommendedActions: [
            "Conduct immediate fogging operations",
            "Deploy additional medical teams",
            "Issue public health advisory"
        ]
    },
    "2": {
        title: "Manila Cluster",
        location: "Manila",
        cases: "87 this week",
        increase: "45%",
        assessment: "This area is approaching the epidemic threshold with moderate increase in cases.",
        updated: "Today, 8:30 AM",
        status: "MODERATE ALERT",
        risk: "moderate",
        recommendedActions: [
            "Increase public awareness campaigns",
            "Schedule neighborhood cleanups",
            "Monitor high-risk areas daily"
        ]
    },
    "3": {
        title: "Makati Monitoring",
        location: "Makati",
        cases: "23 this week",
        increase: "15%",
        assessment: "This area is being monitored for potential outbreak.",
        updated: "Yesterday, 4:15 PM",
        status: "LOW ALERT",
        risk: "low",
        recommendedActions: [
            "Continue routine inspections",
            "Educate residents on prevention",
            "Maintain mosquito control measures"
        ]
    }
};

// City-specific alert recommendations
const cityRecommendations = {
    "Manila": {
        assessment: "This area has exceeded the epidemic threshold with a rapid increase in cases.",
        recommendedActions: [
            "Conduct immediate fogging operations",
            "Deploy additional medical teams",
            "Issue public health advisory"
        ]
    },
    "Quezon City": {
        assessment: "This area is approaching the epidemic threshold with moderate increase in cases.",
        recommendedActions: [
            "Increase public awareness campaigns",
            "Schedule neighborhood cleanups",
            "Monitor high-risk areas daily"
        ]
    },
    // Add all other cities with their default recommendations
    "Caloocan": {
        assessment: "This area is being monitored for potential outbreak.",
        recommendedActions: [
            "Conduct immediate fogging operations",
            "Deploy additional medical teams",
            "Issue public health advisory"
        ]
    },
    // ... other cities
    "Default": {
        assessment: "This area is being monitored for potential outbreak.",
        recommendedActions: [
            "Continue routine inspections",
            "Educate residents on prevention",
            "Maintain mosquito control measures"
        ]
    }
};

function getCityRecommendations(city) {
    return cityRecommendations[city] || cityRecommendations["Default"];
}

function updateCityRecommendations(city, assessment, actions) {
    if (!cityRecommendations[city]) {
        cityRecommendations[city] = {};
    }
    cityRecommendations[city].assessment = assessment;
    cityRecommendations[city].recommendedActions = actions;
    
    // Also update any active alerts for this city
    Object.values(alertData).forEach(alert => {
        if (alert.location === city) {
            alert.assessment = assessment;
            alert.recommendedActions = actions;
        }
    });
}
// Filter alerts by risk level
function filterAlerts(riskLevel) {
    const alerts = document.querySelectorAll('.alert-card');
    alerts.forEach(alert => {
        if (riskLevel === 'all' || alert.dataset.risk === riskLevel) {
            alert.style.display = 'block';
        } else {
            alert.style.display = 'none';
        }
    });
}

// Initialize filter functionality
function initializeFilter() {
    const filterBtn = document.getElementById('filter-btn');
    const riskSelect = document.querySelector('select[name="risk-level"]');
    
    if (filterBtn && riskSelect) {
        filterBtn.addEventListener('click', () => {
            const riskLevel = riskSelect.value;
            filterAlerts(riskLevel);
        });
    }
    
    // Also filter when risk select changes
    if (riskSelect) {
        riskSelect.addEventListener('change', (e) => {
            filterAlerts(e.target.value);
        });
    }
}
// Handle alert details modal
function setupAlertModal() {
    const modal = document.getElementById('alert-modal');
    const closeBtn = document.getElementById('close-modal');
    const detailBtns = document.querySelectorAll('.alert-details-btn');

    detailBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const alertId = btn.getAttribute('data-alert-id');
            const alert = alertData[alertId];
            
            if (alert) {
                document.getElementById('alert-modal-title').textContent = alert.title;
                document.getElementById('alert-location').textContent = alert.location;
                document.getElementById('alert-cases').textContent = alert.cases;
                document.getElementById('alert-increase').textContent = alert.increase;
                document.getElementById('alert-assessment').textContent = alert.assessment;
                document.getElementById('alert-updated').textContent = alert.updated;
                
                // Update status indicator
                const statusIndicator = document.querySelector('#alert-modal .font-bold.text-red-500');
                statusIndicator.textContent = alert.status;
                statusIndicator.previousElementSibling.className = `inline-block w-3 h-3 ${alert.risk === 'high' ? 'bg-red-500' : alert.risk === 'moderate' ? 'bg-yellow-500' : 'bg-green-500'} rounded-full mr-2`;
                
                // Update recommended actions
                const actionsList = document.querySelector('#alert-modal ul');
                actionsList.innerHTML = '';
                alert.recommendedActions.forEach(action => {
                    const li = document.createElement('li');
                    li.className = 'flex items-start';
                    li.innerHTML = `
                        <i data-feather="check-circle" class="text-green-500 mr-2 mt-0.5"></i>
                        <span>${action}</span>
                    `;
                    actionsList.appendChild(li);
                });
                feather.replace();
                
                modal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            }
        });
    });
closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = 'auto';
        }
    });
}
// Populate cases table
function populateCasesTable() {
    const tableBody = document.getElementById('cases-table-body');
    if (!tableBody) return;

    // TODO: Replace this with database fetch in the future
    // Example: const casesData = await fetch('/api/cases').then(r => r.json());
    
    // REMOVE (dummy data for table) - Replace with database fetch
    const casesData = [
        { city: 'Quezon City', cases: 247, risk: 'High', lastUpdated: 'Jan 15, 2025' },
        { city: 'Manila', cases: 128, risk: 'High', lastUpdated: 'Jan 15, 2025' },
        { city: 'Caloocan', cases: 87, risk: 'High', lastUpdated: 'Jan 14, 2025' },
        { city: 'Las Piñas', cases: 65, risk: 'Moderate', lastUpdated: 'Jan 14, 2025' },
        { city: 'Makati', cases: 45, risk: 'Moderate', lastUpdated: 'Jan 14, 2025' },
        { city: 'Malabon', cases: 38, risk: 'Moderate', lastUpdated: 'Jan 13, 2025' },
        { city: 'Mandaluyong', cases: 32, risk: 'Moderate', lastUpdated: 'Jan 13, 2025' },
        { city: 'Marikina', cases: 28, risk: 'Moderate', lastUpdated: 'Jan 13, 2025' },
        { city: 'Muntinlupa', cases: 25, risk: 'Low', lastUpdated: 'Jan 12, 2025' },
        { city: 'Navotas', cases: 22, risk: 'Low', lastUpdated: 'Jan 12, 2025' },
        { city: 'Parañaque', cases: 19, risk: 'Low', lastUpdated: 'Jan 12, 2025' },
        { city: 'Pasay', cases: 16, risk: 'Low', lastUpdated: 'Jan 11, 2025' },
        { city: 'Pasig', cases: 14, risk: 'Low', lastUpdated: 'Jan 11, 2025' },
        { city: 'San Juan', cases: 12, risk: 'Low', lastUpdated: 'Jan 11, 2025' },
        { city: 'Taguig', cases: 10, risk: 'Low', lastUpdated: 'Jan 10, 2025' },
        { city: 'Valenzuela', cases: 8, risk: 'Low', lastUpdated: 'Jan 10, 2025' },
        { city: 'Pateros', cases: 5, risk: 'Low', lastUpdated: 'Jan 10, 2025' }
    ];
    // END REMOVE (dummy data for table)

    // Sort by cases (descending)
    casesData.sort((a, b) => b.cases - a.cases);

    // Clear existing rows
    tableBody.innerHTML = '';

    // Populate table
    casesData.forEach(item => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50';
        
        const riskBadgeClass = {
            'Low': 'bg-green-100 text-green-800',
            'Moderate': 'bg-yellow-100 text-yellow-800',
            'High': 'bg-redorange-100 text-red-800',
            'VeryHigh': 'bg-red-100 text red-800'
        }[item.risk] || 'bg-gray-100 text-gray-800';

        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${item.city}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${item.cases}</td>
            <td class="px-6 py-4 whitespace-nowrap">
                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${riskBadgeClass}">
                    ${item.risk}
                </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${item.lastUpdated}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupAlertModal();
    initializeFilter();
    
    // Populate cases table after a short delay to ensure SVG paths are loaded
    setTimeout(() => {
        populateCasesTable();
    }, 100);
});
