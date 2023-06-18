module.exports = grammar({
	name: 'jspdl',

	rules: {
		program: $ => repeat(choice(
			$._statement_and_declaration,
			$.function_declaration
		)),

		type: $ => field("type", choice(
			'int',
			'boolean',
			'string'
		)),
		_statement_and_declaration: $ => choice(
			$.let_statement,
			$._statement),
		block_and_declaration: $ => seq('{', repeat($._statement_and_declaration), '}'),
		block: $ => seq('{', repeat($._statement), '}'),
		_statement: $ => choice(
			$.if_statement,
			$.do_while_statement,
			$.return_statement,
			$.print_statement,
			$.input_statement,
			seq($.function_call, ';'),
			$.assignment_statement,
			seq($.post_increment_statement, ';')

		),
		let_statement: $ => seq('let', field("type", $.type), field("identifier", $.identifier), ';'),
		if_statement: $ => seq('if', '(', field("if_condition", $._expression), ')', field("if_body", choice($._statement, $.block))),
		do_while_statement: $ =>
			seq('do', field("do_while_body", $.block), 'while', '(', field("do_while_condition", $._expression), ')', ';'),
		return_statement: $ => seq('return', field("return_value", optional($._expression)), ';'),
		print_statement: $ => seq('print', '(', $._expression, ')', ';'),
		input_statement: $ => seq('input', '(', $.identifier, ')', ';'),
		function_call: $ => seq($.identifier, '(', optional($.argument_list), ')'),
		assignment_statement: $ => seq(field("identifier", $.identifier), '=', $._expression, ';'),
		post_increment_statement: $ => seq($.identifier, '++'),

		argument_list: $ => seq($._expression, repeat(seq(',', $._expression))),

		function_declaration: $ => seq(
			'function',
			$.identifier,
			optional($.type),
			$.argument_declaration_list,
			$.block_and_declaration
		),

		argument_declaration_list: $ =>
			seq('(',
				optional($.argument_declaration),
				optional(repeat(seq(',', $.argument_declaration))),
				')'),
		argument_declaration: $ =>
			seq(field('type', $.type),
				field('name', $.identifier),),

		parenthesized_expression: $ => seq(
			'(',
			$._expression,
			')'
		),

		_expression: $ => choice(
			$.value,
			$.or_expression,
			$.equality_expression,
			$.addition_expression
		),


		or_expression: $ => prec.left(1, seq($._expression, '||', $._expression)),    // Logical
		equality_expression: $ => prec.left(2, seq($._expression, '==', $._expression)),    // Equality
		addition_expression: $ => prec.left(3, seq($._expression, '+', $._expression)),     // Addition

		value: $ => choice(
			$._expression_value,
			$.function_call,
			$.parenthesized_expression,
			$.identifier
		),

		_expression_value: $ => choice(
			$.post_increment_statement,
			$.literal_string,
			$.literal_number,
			$.literal_boolean
		),
		literal_string: $ => /"[^"]*"/,
		literal_number: $ => /-?\d+/,
		literal_boolean: $ => choice('true', 'false'),
		identifier: $ => /[a-zA-Z_]\w*/
	},

	extras: $ => [
		/\s/,
		/\n/,
		/\xa0/, // non-breaking space
		/\/\/.*/, // single-line comment
		/\/\*[^*]*\*+(?:[^/*][^*]*\*+)*\//, // multi-line comment
	],

});