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


    // Show the user's original uploaded image
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



    // show annotated image as separate response
    function addAnnotatedImageMessage(file, detections) {
        if (!detections || detections.length === 0) return;

        const message = document.createElement("div");
        message.classList.add("message", "assistant");

        const image = new Image();
        const imageUrl = URL.createObjectURL(file);

        image.onload = () => {
            URL.revokeObjectURL(imageUrl);

            const canvas = document.createElement("canvas");
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            ctx.drawImage(image, 0, 0);

            // draw a box around the highest-confidence candidate
            const detection = detections[0];

            // Keep bounding-box coordinates within the image
            const x1 = Math.max(0, Math.min(detection.x1, canvas.width));
            const y1 = Math.max(0, Math.min(detection.y1, canvas.height));
            const x2 = Math.max(0, Math.min(detection.x2, canvas.width));
            const y2 = Math.max(0, Math.min(detection.y2, canvas.height));

            if (x2 <= x1 || y2 <= y1) return;

            const lineWidth = Math.max(5, Math.round(Math.min(canvas.width, canvas.height) / 200));

            ctx.strokeStyle = "#FF3B30";
            ctx.lineWidth = lineWidth;

            // Draw the outline inside the predicted bounding box
            ctx.strokeRect(
                x1 + lineWidth / 2,
                y1 + lineWidth / 2,
                Math.max(0, x2 - x1 - lineWidth),
                Math.max(0, y2 - y1 - lineWidth)
            );

            // Create an annotated image without modifying the original
            canvas.toBlob((blob) => {
                if (!blob) return;

                const annotatedUrl = URL.createObjectURL(blob);
                const annotatedImage = document.createElement("img");

                annotatedImage.src = annotatedUrl;
                annotatedImage.alt = "Image with possible Waldo location marked";
                annotatedImage.classList.add("chat-image");

                annotatedImage.addEventListener("load", () => {
                    URL.revokeObjectURL(annotatedUrl);
                }, { once: true });

                message.appendChild(annotatedImage);
                messages.scrollTop = messages.scrollHeight;

            }, "image/png");
        };

        image.onerror = () => {
            URL.revokeObjectURL(imageUrl);
            message.textContent = "Couldn't generate the annotated image.";
        };

        image.src = imageUrl;

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
            }

            messageInput.value = "";

            // display search controller's response
            if (hadImage && searchResult) {
                let reply = searchResult.message;

                // show location & confidence if Waldo was detected
                if (searchResult.found && searchResult.detections.length > 0) {
                    const detection = searchResult.detections[0];

                    const centerX = Math.round((detection.x1 + detection.x2) / 2);
                    const centerY = Math.round((detection.y1 + detection.y2) / 2);
                    const confidence = (detection.confidence * 100).toFixed(1);

                    reply += `\nPossible location: (${centerX}, ${centerY})`;
                    reply += `\nConfidence: ${confidence}%`;
                }

                // make sure simulated results get shown differently
                if (searchResult.mock) {
                    reply = `Demo reply: ${reply}`;
                }

                // show number of regions examined
                if (searchResult.regions_examined != null) {
                    reply += `\nRegions examined: ${searchResult.regions_examined}`;
                }

                addMessage(reply, "assistant");

                // send separate annotated image from assistant
                if (searchResult.found && searchResult.detections.length > 0) {
                    addAnnotatedImageMessage(selectedImage, searchResult.detections);
                }
            } else {
                // text-only not implented through search API
                addMessage("Demo reply: Text-based AI chat isn't connected yet", "assistant");
            }

            // clear attachment after showing results
            if (hadImage) {
                clearSelectedImage();
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