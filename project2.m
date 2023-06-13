function project2
    % Display a welcome message
    % rng(55);
    disp(['Welcome to the Feature Selection Algorithm.' newline])

    % Display dataset choices and get user input
    disp(['Select one of the dataset choices:' newline ...
          '1) Small Dataset-1' newline ...
          '2) Large Dataset-16' newline ...
          '3) XLarge Dataset-6' newline ...
          '4) Real-world Dataset' newline ...
          '5) Enter your own Dataset']);
    dataset_choice = input('');

    % Select the datset name based on user choice
    if dataset_choice >= 1 && dataset_choice <= 3
        dataset_list = ["CS170_small_Data__1.txt", ...
                        "CS170_large_Data__16.txt", ...
                        "CS170_XXXlarge_Data__6.txt"];
        file_name = dataset_list(dataset_choice);
    elseif dataset_choice == 4
        file_name = 'student-mat.csv';
    else
        prompt = "Type in the name of the file to test: ";
        file_name = input(prompt, 's');
    end

    % Get the algorithm choice from the user
    disp(['Type in the number of the algorithm you want to run.' newline ...
          9 '1) Forward Selection' newline ...
          9 '2) Backward Elimination '])
    algorithm_choice = input('');
    
    % Load the dataset and clean real-world dataset if necessary
    if dataset_choice ~= 4
        data = load(file_name);
    else
        % Perform the necessary data cleaning
        data = readtable(file_name, 'Delimiter', ';');
        grades = data.G3;
        new_grades = zeros(size(grades));

        % Set the grade thresholds
        low_grades = 10;
        middle_grades = 15;

        % Convert the final grades to a class label with 3 classes
        for i = 1 : length(grades)
            if grades(i) < low_grades
                new_grades(i) = 1;
            elseif grades(i) < middle_grades
                new_grades(i) = 2;
            else
                new_grades(i) = 3;
            end
        end

        % Replace the final grades with the class labels
        data.G3 = new_grades;

        % Remove the categorical data from the dataset
        data = removevars(data, {'school', 'sex', 'address', 'famsize', ...
                                 'Pstatus', 'Mjob', 'Fjob', 'reason', ...
                                 'guardian', 'schoolsup', 'famsup', 'paid', ...
                                 'activities', 'nursery', 'higher', ...
                                 'internet', 'romantic'});
        
        % Normalize the data to be between 0 and 1
        for i = 1 : width(data) - 1
            data{:, i} = (data{:, i} - min(data{:, i})) / ...
                         (max(data{:, i}) - min(data{:, i}));
        end

        % Switch the final grade class label to the first column
        data = [data(:, 'G3') data(:, setdiff(data.Properties.VariableNames, 'G3'))];

        % write the cleaned data to a new file
        writetable(data, 'cleaned-student-mat.csv')
        data = table2array(data);
    end

    % Display the amount of features and instances
    num_features = num2str(size(data, 2) - 1);
    num_instances = size(data, 1);
    disp(['This dataset has ', num2str(num_features), ' features (not' ...
          ' including the class attribute), with ', num2str(num_instances), ...
          ' instances.' newline])

    % Perform the chosen feature search
    if algorithm_choice == 1
        % Display accuracy from k-fold cross validation using all features
        listz = (1:size(data, 2) - 2);
        initial_accuracy = leave_one_out_cross_validation(data, listz, size(data, 2));
        fprintf(['Running nearest neighbor with all ', num2str(num_features), ...
              ' features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of %.1f%%\n'], initial_accuracy * 100)
        
        % Display accuracy from k-fold cross validation using no features
        listz = [];
        initial_accuracy = leave_one_out_cross_validation(data, listz, -1);
        fprintf(['Running nearest neighbor with 0 ', ...
              'features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of %.1f%%\n'], initial_accuracy * 100)
        
        % Begin forward search
        disp(['Beginning search.' newline])
        feature_search(data);
    else
        % Display accuracy from k-fold cross validation using no features
        listz = (1);
        initial_accuracy = backward_leave_one_out_cross_validation(data, listz, 2);
        fprintf(['Running nearest neighbor with 0 ', ...
              'features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of %.1f%%\n'], initial_accuracy * 100)

        % Display accuracy from k-fold cross validation using all features
        listz = (1:size(data, 2) - 1);
        initial_accuracy = backward_leave_one_out_cross_validation(data, listz, 0);
        fprintf(['Running nearest neighbor with all ', num2str(num_features), ...
              ' features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of %.1f%%\n'], initial_accuracy * 100)
        
        % Begin backward elimination
        disp(['Beginning backward search.' newline])
        backward_feature_search(data);
    end
end

% Function to calculate K-fold cross validation
function accuracy = leave_one_out_cross_validation(data, current_set, feature_to_add)
    % Create List with features being used and adjust values by 1
    features = current_set + 1;
    if feature_to_add ~= -1
        features = [features feature_to_add];
    end
    % Take the columns not being considered and set them to zero
    list_features = (2:size(data, 2));
    not_features = setxor(features, list_features);
    data(:, not_features) = 0;

    % Set a threshold to determine if sampling is required
    threshold = 2048;
    if size(data, 1) > threshold
        % Set the amount of data to sample and adjust the data
        sample_size = 0.2;
        total_samples = floor(size(data, 1) * sample_size);
        indices = randperm(size(data, 1), total_samples);
        data = data(indices, :);
    end
   
    number_correctly_classified = 0;

    % Loop through the dataset columns
    for i = 1 : size(data, 1)
        object_to_classify = data(i, 2:end);
        label_object_to_classify = data(i, 1);

        nearest_neighbor_distance = inf;
        nearest_neighbor_location = inf;

        % For each feature, consider distance to every other feature
        for k = 1 : size(data, 1)
            % Compute the euclidean distance for each neighbor except onseself
            if k ~= i
                distance = sqrt(sum((object_to_classify - data(k, 2:end)).^2));

                % Update the nearest neighbor if one is found
                if distance < nearest_neighbor_distance
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k; 
                    nearest_neighbor_label = data(nearest_neighbor_location, 1);
                end
            end
        end

         % Increment the objects classified correctly
        if label_object_to_classify == nearest_neighbor_label
            number_correctly_classified = number_correctly_classified + 1;
        end
    end

    % Calculate the accuracy of the k-fold cross validation
    accuracy = number_correctly_classified / size(data, 1);
end

% Function to search for the best features to add
function feature_search(data)
    % Declare and initialize variables
    current_set_of_features = [];
    best_set_of_features = [];
    best_accuracy = 0;
    warning_displayed = 0;

    % Loop through the levels for each feature
    for i = 1 : size(data, 2) - 1 
        feature_to_add_at_this_level = [];
        best_so_far_accuracy = 0;

        % Consider adding every other feature
        for k = 1 : size(data, 2) - 1
            if isempty(intersect(current_set_of_features, k))
                % calculate the accuracy for the feature being considered
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1);
                printList = regexprep(mat2str(current_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
                if isempty(current_set_of_features)
                    fprintf('\tusing feature(s) {%d%s} accuracy is %.1f%%\n',...
                            k, printList, accuracy * 100)
                else
                    fprintf('\tusing feature(s) {%d,%s} accuracy is %.1f%%\n',...
                            k, printList, accuracy * 100)
                end
                
                % Update the best accuracy and the feature being added
                if accuracy > best_so_far_accuracy
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                end
            end
        end

        % Store the new feature for the search
        if (best_so_far_accuracy < best_accuracy) && (warning_displayed == 0)
            warning_displayed = 1;
            disp([newline '(Warning, Accuracy has decreased! Continuing search' ...
                   ' in case of local maxima)'])
        end
        current_set_of_features(i) = feature_to_add_at_this_level;
        printList = regexprep(mat2str(current_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
        fprintf('\nFeature set {%s} was best, accuracy is %.1f%%\n\n', ...
                printList, best_so_far_accuracy * 100)
        
        % Update best accuracy and feature set
        if best_so_far_accuracy > best_accuracy
            best_accuracy = best_so_far_accuracy;
            best_set_of_features = current_set_of_features;
        end

    end

    % Output the result of the search
    printList = regexprep(mat2str(best_set_of_features), ...
                         {'\[', '\]', '\s+'}, {'', '', ','});
    fprintf(['Finished forward search!! The best feature subset is {%s} ' ...
             'Which had an accuracy of %.1f%%\n'], ...
              printList, best_accuracy * 100)
end

% Function to perform backward elimination
function backward_feature_search(data)
    % Declare and initialize variables
    current_set_of_features = (1:size(data, 2) - 1);
    best_set_of_features = [];
    best_accuracy = 0;
    warning_displayed = 0;

    % Loop through the levels for the features
    for i = 1 : size(data, 2) - 1
        feature_to_remove_at_this_level = [];
        best_so_far_accuracy = 0;

        % Consider removing each feature
        for k = 1 : size (data, 2) - 1
            % Consider Removing the feature only if it is in the current set
            if ~isempty(intersect(current_set_of_features, k))
                % Calculate the accuracy resulting from the removed feature
                accuracy = backward_leave_one_out_cross_validation(data, current_set_of_features, k + 1);
                temp_print_features = current_set_of_features;
                temp_print_features(temp_print_features == k) = [];
                printList = regexprep(mat2str(temp_print_features), {'\[', '\]', '\s+'}, {'', '', ','});
                if isempty(temp_print_features)
                    fprintf('\tusing feature(s) { } accuracy is %.1f%%\n', ...
                            accuracy * 100)
                else
                    fprintf('\tusing feature(s) {%s} accuracy is %.1f%%\n', ...
                            printList, accuracy * 100)
                end

                % Update the best accuracy and the feature being removed
                if accuracy > best_so_far_accuracy
                    best_so_far_accuracy = accuracy;
                    feature_to_remove_at_this_level = k;
                end
            end
        end
        % Remove the feature for the search
        if (best_so_far_accuracy < best_accuracy) && (warning_displayed == 0)
            warning_displayed = 1;
            disp([newline '(Warning, Accuracy has decreased! Continuing search' ...
                   ' in case of local maxima)'])
        end
        current_set_of_features(current_set_of_features == feature_to_remove_at_this_level) = [];
        printList = regexprep(mat2str(current_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
        if isempty(current_set_of_features)
            fprintf('\nFeature set { } was best, accuracy is %.1f%%\n', ...
                    best_so_far_accuracy * 100)
        else
            fprintf('\nFeature set {%s} was best, accuracy is %.1f%%\n', ...
                    printList, best_so_far_accuracy * 100)
        end

        % Update best accuracy and feature set
        if best_so_far_accuracy > best_accuracy
            best_accuracy = best_so_far_accuracy;
            best_set_of_features = current_set_of_features;
        end
    end

    % Output the result of the search
    printList = regexprep(mat2str(best_set_of_features), ...
                         {'\[', '\]', '\s+'}, {'', '', ','});
    fprintf(['Finished backward search!! The best feature subset is {%s} ' ...
             'Which had an accuracy of %.1f%%\n'], ...
              printList, best_accuracy * 100)
end     

function accuracy = backward_leave_one_out_cross_validation(data, current_set, feature_to_remove)
    % Create a list with the features being used and adjust values by 1
    features = current_set + 1;
    features(features == feature_to_remove) = [];

    % Take the columns not being considered and set them to zero
    list_features = (2:size(data, 2));
    not_features = setxor(features, list_features);
    data(:, not_features) = 0;
    
    number_correctly_classified = 0;

    % Loop through the dataset columns
    for i = 1 : size(data, 1)
        object_to_classify = data(i, 2:end);
        label_object_to_classify = data(i, 1);

        nearest_neighbor_distance = inf;
        nearest_neighbor_location = inf;

        % For each feature, consider distance to every other feature
        for k = 1 : size(data, 1)
            % Compute the euclidean distance for each neighbor except onseself
            if k ~= i
                distance = sqrt(sum((object_to_classify - data(k, 2:end)).^2));

                % Update the nearest neighbor if one is found
                if distance < nearest_neighbor_distance
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k; 
                    nearest_neighbor_label = data(nearest_neighbor_location, 1);
                end
            end
        end

        % Increment the objects classified correctly
        if label_object_to_classify == nearest_neighbor_label
            number_correctly_classified = number_correctly_classified + 1;
        end
    end

    % Calculate the accuracy of the k-fold cross validation
    accuracy = number_correctly_classified / size(data, 1);
end
