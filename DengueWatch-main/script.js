let csvUploaded = false;
const landAreaToCity = {
    24.98: "MANILA CITY",
    171.71: "QUEZON CITY",
    55.8: "CALOOCAN CITY",
    32.69: "LAS PINAS CITY",
    21.57: "MAKATI CITY",
    15.71: "MALABON CITY",
    9.29: "MANDALUYONG CITY",
    21.52: "MARIKINA CITY",
    39.75: "MUNTINLUPA CITY",
    8.94: "NAVOTAS CITY",
    46.57: "PARANAQUE CITY",
    13.97: "PASAY CITY",
    48.46: "PASIG CITY",
    10.4: "PATEROS",
    5.95: "SAN JUAN CITY",
    45.21: "TAGUIG CITY",
    47.02: "VALENZUELA CITY"
};

const alertStyles = {
    Low: {
        border: "border-green-500",
        badge: "bg-green-100 text-green-800",
        dot: "bg-green-500",
        label: "LOW ALERT"
    },
    Moderate: {
        border: "border-yellow-500",
        badge: "bg-yellow-100 text-yellow-800",
        dot: "bg-yellow-500",
        label: "MEDIUM ALERT"
    },
    High: {
        border: "border-red-500",
        badge: "bg-red-100 text-red-800",
        dot: "bg-red-500",
        label: "HIGH ALERT"
    },
    "Very High": {
        border: "border-red-500",
        badge: "bg-red-100 text-red-800",
        dot: "bg-red-500",
        label: "CRITICAL ALERT"
    }
};


const fileInput = document.getElementById("csv-upload");
let selectedFile = null;


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

function renderRiskChart() {
    if (!document.getElementById('riskChart')) return;

    const riskCtx = document.getElementById('riskChart').getContext('2d');

    const paths = document.querySelectorAll('.risk-path');
    let riskCounts = { Low: 0, Moderate: 0, High: 0, VeryHigh: 0 };

    paths.forEach(path => {
        const risk = path.dataset.risk;
        if (riskCounts[risk] !== undefined) {
            riskCounts[risk]++;
        }
    });

    if (window.riskChartInstance) {
        window.riskChartInstance.destroy();
    }

    window.riskChartInstance = new Chart(riskCtx, {
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
                backgroundColor: ['#4ade80', '#fbbf24', '#f87171', '#dc2626'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

async function updateMapRisks() {
    try {
        const response = await fetch('http://localhost:8000/api/latest-forecast');
        if (!response.ok) throw new Error("Failed to fetch risk data");

        const riskDataFromDB = await response.json();

        document.querySelectorAll('.risk-path').forEach(path => {
            const city = path.getAttribute('city');
            const record = riskDataFromDB.find(item => item.city === city);
            const risk = record ? record.risk_level : null;

            path.setAttribute('risk_level', risk || '');
            path.dataset.risk = risk || '';
            path.style.fill = riskColors[risk] || '#bfbfbf';
        });

        renderRiskChart();

    } catch (err) {
        console.error('Error fetching or applying risk data:', err);
    }
}



// Main initialization function
async function initializeApp() {
// Load required scripts based on current page
    await Promise.all([
        loadScript('https://cdn.jsdelivr.net/npm/chart.js'),
        loadScript('https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js'),
        loadScript('https://cdn.jsdelivr.net/npm/animejs@3.2.1/lib/anime.min.js'),
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
    // Create or update risk chart if element exists
    
        // Animate elements
        anime({
            targets: '.risk-high',
            scale: [1, 1.1, 1],
            duration: 1500,
            loop: true,
            easing: 'easeInOutSine'
        });

    // Admin page functionality
    if (document.getElementById('dropzone')) {
        const dropzone = document.getElementById("dropzone");
        const submitBtn = document.getElementById("submit-btn");
        const uploadArea = document.getElementById("upload-area");
        const processingArea = document.getElementById("processing-area");
        const successMessage = document.getElementById("success-message");

        // Drag & drop
        ["dragenter", "dragover"].forEach(evt => {
            dropzone.addEventListener(evt, e => {
                e.preventDefault();
                dropzone.classList.add("border-blue-500");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            dropzone.addEventListener(evt, e => {
                e.preventDefault();
                dropzone.classList.remove("border-blue-500");
            });
        });

        dropzone.addEventListener("drop", e => {
            const files = e.dataTransfer.files;
            if (files.length) {
                selectedFile = files[0];
                fileInput.files = files;
                previewLocalCSV(selectedFile);
            }
        });

        fileInput.addEventListener("change", e => {
            selectedFile = e.target.files[0];
        });

        // Submit → FastAPI
        submitBtn.addEventListener("click", async () => {
            if (!selectedFile) {
                alert("Please upload a CSV file");
                return;
            }

            uploadArea.classList.add("hidden");
            processingArea.classList.remove("hidden");

            const formData = new FormData();
            formData.append("file", selectedFile);

            try {
                const res = await fetch(
                    "http://localhost:8000/preprocessing?weeks_ahead=1",
                    { method: "POST", body: formData }
                );

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Upload failed");
                }

                await res.json();

                processingArea.classList.add("hidden");
                successMessage.classList.remove("hidden");

                setTimeout(() => {
                    successMessage.classList.add("hidden");
                    uploadArea.classList.remove("hidden");
                }, 2000);

            } catch (err) {
                processingArea.classList.add("hidden");
                uploadArea.classList.remove("hidden");
                alert(err.message);
            }
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
                const name = path.getAttribute('city') || 'Unknown Area';
                const risk = path.getAttribute('risk_level') || 'Low';

                mapCard.innerHTML = `
                    <div style="font-weight:600;">${name}</div>
                    <div style="font-size:12px; margin-top:4px;">
                        Risk Level: <strong>${risk}</strong><br>
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

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        previewLocalCSV(file);
    });

}

window.addEventListener('DOMContentLoaded', updateMapRisks);

async function loadAlertsFromAPI() {
    const container = document.getElementById("alerts-container");
    if (!container) return;

    container.innerHTML = "<p class='text-gray-500'>Loading alerts...</p>";

    try {
        const res = await fetch("http://localhost:8000/api/alerts");
        if (!res.ok) throw new Error("Failed to fetch alerts");

        const alerts = await res.json();
        container.innerHTML = "";

        if (!alerts.length) {
            container.innerHTML = "<p class='text-gray-500'>No alerts available.</p>";
            return;
        }

        alerts.forEach((alert, index) => {
            const style = alertStyles[alert.alert_level] || alertStyles.Low;

            const card = document.createElement("div");
            card.className = `
                alert-card bg-white rounded-xl shadow-lg overflow-hidden
                border-l-4 ${style.border}
            `;
            card.dataset.risk = alert.alert_level.toLowerCase();

            card.innerHTML = `
                <div class="p-6">
                    <div class="flex justify-between items-start">
                        <div>
                            <div class="flex items-center mb-2">
                                <span class="inline-block w-3 h-3 ${style.dot} rounded-full mr-2"></span>
                                <span class="font-bold">${style.label}</span>
                            </div>
                            <h2 class="text-xl font-bold">${alert.city}</h2>
                            <p class="text-gray-600">
                                ${alert.percent_change > 0 ? "Increase" : "Decrease"}
                                of ${Math.abs(alert.percent_change)}% from last week
                            </p>
                        </div>
                        <span class="px-3 py-1 rounded-full text-sm font-medium ${style.badge}">
                            ${alert.alert_level}
                        </span>
                    </div>

                    <div class="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                            <p class="text-gray-500">Current Week</p>
                            <p class="font-semibold">${alert.current_week_cases}</p>
                        </div>
                        <div>
                            <p class="text-gray-500">Previous Week</p>
                            <p class="font-semibold">${alert.previous_week_cases}</p>
                        </div>
                        <div>
                            <p class="text-gray-500">Last Updated</p>
                            <p class="font-semibold">${alert.last_updated}</p>
                        </div>
                    </div>
                </div>

                <div class="bg-gray-50 px-6 py-3 flex justify-end">
                    <button
                        class="text-blue-600 hover:text-blue-700 flex items-center alert-details-btn"
                        data-alert='${encodeURIComponent(JSON.stringify(alert))}'>
                        <span>More Details</span>
                        <i data-feather="chevron-right" class="w-4 h-4 ml-1"></i>
                    </button>

                </div>
            `;

            container.appendChild(card);
        });

        feather.replace();
        setupAlertModalDynamic();

    } catch (err) {
        console.error(err);
        container.innerHTML = "<p class='text-red-500'>Failed to load alerts.</p>";
    }
}

function setupAlertModalDynamic() {
    const modal = document.getElementById('alert-modal');
    const closeBtn = document.getElementById('close-modal');
    const detailBtns = document.querySelectorAll('.alert-details-btn');

    detailBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            // Decode and parse the alert object from data-alert
            const alert = JSON.parse(decodeURIComponent(btn.dataset.alert));

            // Fetch latest risk level from API for this city
            let riskLevel = "LOW";
            try {
                const res = await fetch("http://localhost:8000/api/risk-latest");
                if (res.ok) {
                    const latestRisks = await res.json();
                    const cityRisk = latestRisks.find(r => r.city === alert.city);
                    if (cityRisk && cityRisk.risk_level) {
                        riskLevel = cityRisk.risk_level;
                    }
                }
            } catch (e) {
                console.error("Failed to fetch latest risk:", e);
            }

            // Populate modal content
            document.getElementById('alert-modal-title').textContent = alert.city;
            document.getElementById('alert-location').textContent = alert.city;
            document.getElementById('alert-cases').textContent = alert.current_week_cases;
            document.getElementById('alert-increase').textContent = alert.percent_change + '%';
            document.getElementById('alert-assessment').textContent =
                `This area is classified as ${riskLevel} risk.`;
            document.getElementById('alert-updated').textContent = alert.last_updated;

            // Update status indicator (dot + text)
            const statusIndicator = document.querySelector('#alert-modal .font-bold');
            if (statusIndicator) {
                statusIndicator.textContent = riskLevel;
                statusIndicator.className = `font-bold ${
                    riskLevel === 'CRITICAL' ? 'text-red-500' :
                    riskLevel === 'MEDIUM' ? 'text-yellow-600' : 'text-green-600'
                }`;

                const dot = statusIndicator.previousElementSibling;
                if (dot) {
                    dot.className = `inline-block w-3 h-3 ${
                        riskLevel === 'CRITICAL' ? 'bg-red-500' :
                        riskLevel === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                    } rounded-full mr-2`;
                }
            }

            // Recommended actions
            const actionsList = document.querySelector('#alert-modal ul');
            actionsList.innerHTML = '';
            const recommended = alert.recommendedActions || [];
            recommended.forEach(action => {
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



// Weather update function
function updateWeather(city = "Manila") {
    const apiKey = "fd5f51d2101b472aa31195301262101"; // Replace with your WeatherAPI key
    fetch(`https://api.weatherapi.com/v1/forecast.json?key=${apiKey}&q=${encodeURIComponent(city)}&days=14`)
        .then(response => response.json())
        .then(data => {
            // Current weather
            const temperature = Math.round(data.current.temp_c);
            const rainProbability = data.forecast.forecastday[0].day.daily_chance_of_rain + "%";

            // Risk calculation
           
            let advisory = "Normal weather conditions. Maintain regular prevention measures.";
            const rainChance = parseInt(data.forecast.forecastday[0].day.daily_chance_of_rain);

            if (rainChance > 70) {
                advisory = "Very high rain probability increases standing water. Expect increased mosquito activity.";
            } else if (rainChance > 50) {
                advisory = "Frequent rain expected. Check and eliminate standing water around your area.";
            } else if (rainChance > 30) {
                advisory = "Moderate rain probability. Stay alert and continue preventive actions.";
            }

            // Update current weather UI
            document.getElementById('rain-probability').textContent = rainProbability;
            document.getElementById('temperature').textContent = `${temperature}°C`;
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

function updateAlertModalSeverity(alertLevel) {
    const normalizedLevel = alertLevel?.trim() || "Low";

    const levelConfig = {
        "Very High": {
            dot: "bg-red-600",
            text: "text-red-600",
            label: "CRITICAL ALERT",
            riskLabel: "Critical Risk Area",
            bg: "bg-red-50",
            icon: "text-red-500"
        },
        High: {
            dot: "bg-red-500",
            text: "text-red-500",
            label: "HIGH ALERT",
            riskLabel: "High Risk Area",
            bg: "bg-red-50",
            icon: "text-red-500"
        },
        Moderate: {
            dot: "bg-yellow-500",
            text: "text-yellow-600",
            label: "MODERATE ALERT",
            riskLabel: "Moderate Risk Area",
            bg: "bg-yellow-50",
            icon: "text-yellow-500"
        },
        Low: {
            dot: "bg-green-500",
            text: "text-green-600",
            label: "LOW ALERT",
            riskLabel: "Low Risk Area",
            bg: "bg-green-50",
            icon: "text-green-500"
        }
    };

    const config = levelConfig[normalizedLevel] || levelConfig.Low;

    document.getElementById("alert-severity-badge").innerHTML = `
        <span class="inline-block w-3 h-3 ${config.dot} rounded-full mr-2"></span>
        <span class="font-bold ${config.text}">${config.label}</span>
    `;

    document.getElementById("alert-risk-header").innerHTML = `
        <i data-feather="alert-triangle" class="${config.icon} mr-2"></i>
        <span class="font-bold ${config.text}">${config.riskLabel}</span>
    `;

    document
        .getElementById("alert-risk-header")
        .parentElement
        .className = `${config.bg} p-4 rounded-lg`;

    feather.replace();
}

function setupAlertModalDynamic() {
    const modal = document.getElementById('alert-modal');
    const closeBtn = document.getElementById('close-modal');
    const detailBtns = document.querySelectorAll('.alert-details-btn');

    detailBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Decode and parse alert object
            const alert = JSON.parse(decodeURIComponent(btn.dataset.alert));

            // Normalize risk level (backend may send VeryHigh)
            const normalizedLevel =
                alert.alert_level === "VeryHigh"
                    ? "Very High"
                    : alert.alert_level;

            // Populate modal title and basic info
            document.getElementById('alert-modal-title').textContent = alert.city;
            document.getElementById('alert-location').textContent = alert.city;
            document.getElementById('alert-cases').textContent = alert.current_week_cases;
            document.getElementById('alert-increase').textContent = alert.percent_change + '%';
            document.getElementById('alert-updated').textContent = alert.last_updated;
            // Update severity badge
            updateAlertModalSeverity(normalizedLevel);

            // Use risk_assessment from DB (fallback text if missing)
            document.getElementById('alert-assessment').textContent =
                alert.risk_assessment ||
                `This area is classified as ${normalizedLevel} risk.`;

            // Populate risk header (icon + text)
            const riskHeader = document.getElementById('alert-risk-header');
            if (riskHeader) {
                const levelColors = {
                    "Very High": {
                        text: 'text-red-600',
                        icon: 'text-red-500',
                        label: 'High Risk Area'
                    },
                    High: {
                        text: 'text-red-600',
                        icon: 'text-red-500',
                        label: 'High Risk Area'
                    },
                    Moderate: {
                        text: 'text-yellow-600',
                        icon: 'text-yellow-500',
                        label: 'Moderate Risk Area'
                    },
                    Low: {
                        text: 'text-green-600',
                        icon: 'text-green-500',
                        label: 'Low Risk Area'
                    }
                };

                const colors = levelColors[normalizedLevel] || levelColors.Low;

                riskHeader.innerHTML = `
                    <i data-feather="alert-triangle" class="${colors.icon} mr-2"></i>
                    <span class="font-bold ${colors.text}">${colors.label}</span>
                `;
            }

            // Populate recommended actions (from MongoDB)
            const actionsList = document.querySelector('#alert-modal ul');
            actionsList.innerHTML = '';

            if (alert.actions && alert.actions.length > 0) {
                alert.actions.forEach(action => {
                    const li = document.createElement('li');
                    li.className = 'flex items-start';
                    li.innerHTML = `
                        <i data-feather="check-circle" class="text-green-500 mr-2 mt-0.5"></i>
                        <span>${action}</span>
                    `;
                    actionsList.appendChild(li);
                });
            } else {
                actionsList.innerHTML = `<li>No recommended actions.</li>`;
            }

            // Render feather icons
            feather.replace();

            // Show modal
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        });
    });

    // Close modal button
    closeBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    });

    // Close modal when clicking outside
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

async function loadCSVPreview() {
    const tbody = document.getElementById('preview-table');

    try {
        const res = await fetch('http://localhost:8000/api/raw-csv-data');
        const { data } = await res.json();

        tbody.innerHTML = '';

        if (!data.length) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-gray-500 py-4">
                        No data uploaded yet
                    </td>
                </tr>`;
            return;
        }

        data.forEach(row => {
            const landArea = Number(row["LAND AREA"]);
            const city = landAreaToCity[landArea] || "Unknown City";

            const date = `${row["YEAR"]}-${String(row["MONTH"]).padStart(2, "0")}-${String(row["DAY"]).padStart(2, "0")}`;

            tbody.innerHTML += `
                <tr>
                    <td class="px-6 py-2 font-medium">${city}</td>
                    <td class="px-6 py-2">${Number(row["CASES"]).toFixed(2)}</td>
                    <td class="px-6 py-2">${date}</td>
                </tr>
            `;
        });


    } catch (e) {
        console.error("Preview load failed", e);
    }
}

function previewLocalCSV(file) {
    const reader = new FileReader();
    const tbody = document.getElementById('preview-table');

    reader.onload = (evt) => {
        const text = evt.target.result;
        const lines = text.trim().split('\n');

        if (lines.length <= 1) {
            tbody.innerHTML = `<tr><td colspan="3" class="text-center text-gray-500 py-4">No data in file</td></tr>`;
            return;
        }

        const headers = lines[0].split(',').map(h => h.trim());
        const dataRows = lines.slice(1);

        tbody.innerHTML = '';

        dataRows.forEach(line => {
            const cols = line.split(',').map(c => c.trim());
            const row = {};
            headers.forEach((h, i) => row[h] = cols[i]);

            const landArea = Number(row["LAND AREA"]);
            const city = landAreaToCity[landArea] || "Unknown City";
            const date = `${row["YEAR"]}-${String(row["MONTH"]).padStart(2, "0")}-${String(row["DAY"]).padStart(2, "0")}`;
            const cases = Number(row["CASES"]).toFixed(2);

            tbody.innerHTML += `
                <tr>
                    <td class="px-6 py-2 font-medium">${city}</td>
                    <td class="px-6 py-2">${cases}</td>
                    <td class="px-6 py-2">${date}</td>
                </tr>
            `;
        });
    };

    reader.readAsText(file);
}


// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    loadAlertsFromAPI();
    setupAlertModal();
    initializeFilter();
    
    // Populate cases table after a short delay to ensure SVG paths are loaded
    setTimeout(() => {
        populateCasesTable();
    }, 100);
});


