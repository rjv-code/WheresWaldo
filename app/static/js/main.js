
document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    const themeButton = document.getElementById("theme-toggle");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const messages = document.getElementById("chat-messages");
    const newChatButton = document.getElementById("new-chat");
    const welcome = document.getElementById("welcome");

    // default to dark or load theme
    let savedTheme = "dark";

    try {
        savedTheme = localStorage.getItem("waldo-theme") || "dark";
    } catch (error) {
        console.warn("Couldn't load theme...");
    }

    function setTheme(theme) {
        root.dataset.theme = theme;

        themeButton.textContent = theme === "dark"
            ? "Switch to light mode"
            : "Switch to dark mode";

        try {
            localStorage.setItem("waldo-theme", theme);
        } catch (error) {
            console.warn("Couldn't save theme...");
        }
    }

    setTheme(savedTheme === "light" ? "light" : "dark");

    themeButton.addEventListener("click", () => {
        setTheme(root.dataset.theme === "dark" ? "light" : "dark");
    });

    // Add a message to the conversation
    function addMessage(text, sender) {
        const message = document.createElement("div");
        message.classList.add("message", sender);

        // prevent text being parsed as HTML
        message.textContent = text;

        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }

    // handle chat form
    chatForm.addEventListener("submit", (event) => {
        event.preventDefault();

        const text = messageInput.value.trim();

        if (!text) return;

        welcome.hidden = true;
        welcome.style.display = "none";

        addMessage(text, "user");
        messageInput.value = "";

        // Placeholder until the AI backend is connected
        addMessage("Demo reply: " + "AI detection is not connected yet.", "assistant");
    });

    // Allow Enter to send, Shift + Enter for a new line
    messageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            chatForm.requestSubmit();
        }
    });

    // Reset the current conversation
    newChatButton.addEventListener("click", () => {
        messages.replaceChildren(welcome);

        welcome.hidden = false;
        welcome.style.display = "";

        messageInput.value = "";
        messageInput.focus();
    });
});