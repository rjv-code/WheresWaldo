document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    const themeButton = document.getElementById("theme-toggle");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const messages = document.getElementById("chat-messages");
    const newChatButton = document.getElementById("new-chat");
    const welcome = document.getElementById("welcome");
    const attachButton = document.getElementById("attach-btn");
    const imagePreview = document.getElementById("image-preview");
    const previewImage = document.getElementById("preview-image");
    const previewFilename = document.getElementById("preview-filename");
    const removeImageButton = document.getElementById("remove-image");
    const uploadError = document.getElementById("upload-error");
    const imageInput = document.getElementById("image-input");

    let selectedImage = null;
    let previewUrl = null;

    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/webp"
    ];

    const maxImageSize = 10485760; // 10 MB

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

    function showUploadError(message) {
        uploadError.textContent = message;
        uploadError.hidden = false;
    }

    function clearUploadError() {
        uploadError.textContent = "";
        uploadError.hidden = true;
    }

    function clearSelectedImage() {
        selectedImage = null;
        imageInput.value = "";

        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }

        previewImage.removeAttribute("src");
        previewFilename.textContent = "";
        imagePreview.hidden = true;
        clearUploadError();
    }

    attachButton.addEventListener("click", () => {
        imageInput.click();
    });


    // Handle image selection and display its preview
    imageInput.addEventListener("change", () => {
        const file = imageInput.files[0];

        // User cancelled the file picker
        if (!file) return;

        // Clear any previous attachment
        clearSelectedImage();

        // Validate image format
        if (!allowedTypes.includes(file.type)) {
            showUploadError(
                "Please select a PNG, JPEG or WebP image."
            );
            return;
        }

        // Validate image size
        if (file.size > maxImageSize) {
            showUploadError(
                "The image must be 10 MB or smaller."
            );
            return;
        }

        // Store the selected file
        selectedImage = file;

        // Create a temporary URL for the browser preview
        previewUrl = URL.createObjectURL(file);

        // Display the image and its filename
        previewImage.src = previewUrl;
        previewFilename.textContent = file.name;
        imagePreview.hidden = false;
    });

    // Allow users to remove their selected image
    removeImageButton.addEventListener("click", () => {
        clearSelectedImage();
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


    // Show uploaded image in the conversation
    function addImageMessage(file) {
        const message = document.createElement("div");
        message.classList.add("message", "user");

        const image = document.createElement("img");
        const imageUrl = URL.createObjectURL(file);

        image.src = imageUrl;
        image.alt = file.name;
        image.classList.add("chat-image");

        image.addEventListener("load", () => {
            URL.revokeObjectURL(imageUrl);
        }, {once: true});

        message.appendChild(image);
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }

    // handle chat form
    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const text = messageInput.value.trim();

        // need to have either text or an image
        if (!text && !selectedImage) return;

        const hadImage = Boolean(selectedImage);

        // upload image if one was selected
        if (selectedImage) {
            const formData = new FormData();
            formData.append("image", selectedImage);

            try {
                const response = await fetch("/api/upload", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (!response.ok) {
                    showUploadError(result.error || "Image upload failed.");
                    return;
                }
            } catch(error) {
                showUploadError("Couldn't connect to the upload server.");
                return;
            }
        }

        // hide welcome screen
        welcome.hidden = true;
        welcome.style.display = "none";

        // show user's message
        if (text) {
            addMessage(text, "user");
        }

        // show preview of uploaded image
        if (selectedImage) {
            addImageMessage(selectedImage);
            clearSelectedImage();
        }

        messageInput.value = "";

        // Placeholder until the AI backend is connected
        addMessage(hadImage ? "Demo reply: Image received. AI detection isn't connected yet." : "Demo reply: AI detection isn't connected yet.", "assistant");
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
        clearSelectedImage();
        messageInput.focus();
    });
});