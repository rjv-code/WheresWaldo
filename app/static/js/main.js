document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    const themeButton = document.getElementById("theme-toggle");
    const chatForm = document.getElementById("chat-form");
    const sendButton = document.querySelector(".send-btn");
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
    let isSubmitting = false;

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

    function setSubmitting(submitting){
        isSubmitting = submitting;

        // disable sending and image changes while processing
        sendButton.disabled = submitting;
        attachButton.disabled = submitting;
        removeImageButton.disabled = submitting;
        imageInput.disabled = submitting;
        newChatButton.disabled = submitting;
        messageInput.readOnly = submitting;

        // update button appearance
        sendButton.textContent = submitting ? "..." : "↑";
        sendButton.setAttribute("aria-label", submitting ? "Processing image" : "Send message");
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

        // prevent duplicate submissions
        if (isSubmitting) return;

        const text = messageInput.value.trim();

        // need to have either text or an image
        if (!text && !selectedImage) return;

        const hadImage = Boolean(selectedImage);
        let searchResult = null;

        clearUploadError();
        setSubmitting(true);

        // upload image if one was selected
        try {
            // send images through the Flask search API
            if (selectedImage) {
                const formData = new FormData();
                formData.append("image", selectedImage);

                let response;

                try {
                    response = await fetch("/api/search", {
                        method: "POST",
                        body: formData
                    });
                } catch(error) {
                    throw new Error("Couldn't connect to the search server.");
                }

                // try decoding server's JSON response
                let result;

                try {
                    result = await response.json();
                } catch (error) {
                    throw new Error("Server returned invalid response.");
                }

                // handle unsuccessful HTTP requests
                if (!response.ok || result.success === false) {
                    throw new Error(result.message || "Image search failed.");
                }

                // store successful search result
                searchResult = result;
            }

            // hide welcome screen after processing
            welcome.hidden = true;
            welcome.style.display = "none";

            // display user's text
            if (text)
                addMessage(text, "user");

            // display uploaded image
            if (selectedImage) {
                addImageMessage(selectedImage);
                clearSelectedImage();
            }

            messageInput.value = "";

            // display search controller's response
            if (hadImage && searchResult) {
                const reply = searchResult.mock ? `Demo reply: ${searchResult.message}` : searchResult.message;

                addMessage(reply, "assistant");
            } else {
                // text-only not implented through search API
                addMessage("Demo reply: Text-based AI chat isn't connected yet", "assistant");
            }
        } catch (error) {
            // preserve image & text in case we need to retry
            showUploadError(error.message || "An unexpected error occurred.");

            console.error("Chat submission failed:", error);
        } finally {
            // restore form controls
            setSubmitting(false);
        }
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